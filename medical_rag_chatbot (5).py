
import os
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from gtts import gTTS
import requests
from typing import Optional, List, Any

INDEX_PATH = "chroma_db"  # مجلد حفظ قاعدة بيانات Chroma
EXCEL_PATH = "Book3.xlsx"

# ===== تحميل بيانات الأدوية والكلمات المفتاحية =====
@st.cache_data
def load_drugs_data(excel_path):
    try:
        drugs_df = pd.read_excel(excel_path, sheet_name="Drugs", header=0)
        drugs_df.columns = drugs_df.columns.str.strip().str.lower()
        keywords_df = pd.read_excel(excel_path, sheet_name="Keywords", header=1)
        keywords_df.columns = keywords_df.columns.str.strip().str.lower()
        display_drugs_df = drugs_df.copy()
        drugs_df["drug"] = drugs_df["drug"].astype(str).str.strip().str.lower()
        keywords_df["keyword"] = keywords_df["keyword"].astype(str).str.strip().str.lower()
        keywords_df["drug"] = keywords_df["drug"].astype(str).str.strip().str.lower()
        return drugs_df, keywords_df, display_drugs_df
    except Exception as e:
        st.error(f"❌ خطأ في قراءة ملف الإكسيل: {e}")
        return None, None, None

# ===== البحث في Excel =====
def search_in_excel(query, drugs_df, keywords_df):
    query = query.lower().strip()
    if drugs_df is not None and "drug" in drugs_df.columns:
        match = drugs_df[drugs_df["drug"].str.contains(query, na=False)]
        if not match.empty:
            return "drug", match
    if keywords_df is not None and {"keyword","drug"}.issubset(keywords_df.columns):
        kw_match = keywords_df[keywords_df["keyword"].str.contains(query, na=False)]
        if not kw_match.empty:
            return "keyword", kw_match.drop(columns=["keyword"], errors="ignore")
    return None, None

# ===== تحميل وتقطيع ملفات PDF =====
def load_medical_docs(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# ===== تضمين النصوص =====
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

def embed_documents(docs):
    embed_model = get_embeddings()
    return Chroma.from_documents(docs, embedding=embed_model, persist_directory=INDEX_PATH)

# ===== حفظ واسترجاع قاعدة البيانات =====
def save_index(index):
    index.persist()  # Chroma يخزن البيانات داخليًا

@st.cache_resource
def load_index():
    if os.path.exists(INDEX_PATH):
        return Chroma(persist_directory=INDEX_PATH, embedding_function=get_embeddings())
    return None

def update_index(new_docs):
    index = load_index()
    if index:
        index.add_documents(new_docs)
        index.persist()
    else:
        index = embed_documents(new_docs)
    return index

# ===== Groq LLM مخصص =====
class GroqLLM(LLM):
    api_key: str
    model_name: str = "llama3-8b-8192"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {"messages": [{"role": "user", "content": prompt}],
                    "model": self.model_name, "max_tokens": 500, "temperature": 0.3}
            response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                     headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return "⚠️ حدث خطأ في الاتصال بـ Groq API"
        except Exception as e:
            return f"⚠️ خطأ: {str(e)}"

# ===== إنشاء نظام الأسئلة والأجوبة =====
@st.cache_resource
def build_qa_system(chroma_index):
    retriever = chroma_index.as_retriever(search_type="similarity", k=4)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ يرجى إضافة GROQ_API_KEY في متغيرات البيئة")
        return None
    llm = GroqLLM(api_key=groq_api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ===== التطبيق الرئيسي =====
def main():
    st.set_page_config(page_title="CHATYMEDx", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #cba37d;'>CHATYMEDx</h1>", unsafe_allow_html=True)

    # تحميل بيانات الأدوية من Excel
    if os.path.exists(EXCEL_PATH):
        drugs_df, keywords_df, display_drugs_df = load_drugs_data(EXCEL_PATH)
    else:
        st.warning("⚠️ لم يتم العثور على ملف الأدوية Excel.")
        drugs_df, keywords_df, display_drugs_df = None, None, None

    # الشريط الجانبي لتحميل الملفات
    with st.sidebar:
        st.markdown("### 📘 Upload your PDF")
        pdf_file = st.file_uploader("PDF File", type=["pdf"])
        st.markdown("### 🖼️ Upload your image:")
        image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

    # التعامل مع PDF
    if pdf_file:
        with st.spinner("📄 Loading and embedding the file..."):
            with open("temp_medical.pdf", "wb") as f:
                f.write(pdf_file.read())
            docs = load_medical_docs("temp_medical.pdf")
            index = update_index(docs)
            st.session_state.qa_chain = build_qa_system(index)
            st.success("✅ File embedded and added to memory.")

    # التعامل مع صورة (ميزة استخراج النص معطلة)
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="The uploaded image", use_container_width=True)
        st.info("📝 ميزة استخراج النص من الصور ستكون متاحة قريباً")

    # إدخال اسم الدواء أو كلمة مفتاحية
    query = st.text_input("Write drug name:")
    if query:
        if drugs_df is not None and keywords_df is not None:
            kind, result_df = search_in_excel(query, drugs_df, keywords_df)
            if kind and result_df is not None and not result_df.empty:
                if kind == "drug":
                    st.success(f"✅ Found drug: {query}")
                else:
                    st.success(f"✅ Found related drug(s) for your keyword")
                st.dataframe(result_df)
            else:
                if "qa_chain" in st.session_state:
                    result = st.session_state.qa_chain.run(query)
                    st.markdown(f"### Your answer: {result}")
                else:
                    st.warning("⚠️ يرجى رفع ملف PDF طبي أولاً أو إضافة GROQ_API_KEY.")

if __name__ == "__main__":
    main()


