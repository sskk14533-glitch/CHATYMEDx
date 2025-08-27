
import os
import streamlit as st
import pickle
import pytesseract
from PIL import Image
from langdetect import detect
import pandas as pd
from docx import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from gtts import gTTS

INDEX_PATH = "faiss_index.pkl"
EXCEL_PATH = "Book3.xlsx"

# ===== تحميل بيانات الأدوية والكلمات المفتاحية من Excel =====
def load_drugs_data(excel_path):
    try:
        # شيت الأدوية
        drugs_df = pd.read_excel(excel_path, sheet_name="Drugs", header=0)
        drugs_df.columns = drugs_df.columns.str.strip().str.lower()
        drugs_df = drugs_df.loc[:, ~drugs_df.columns.str.contains('^unnamed', case=False)]  # تنظيف Unnamed

        if "drug" not in drugs_df.columns:
            st.error(f"❌ مفيش عمود 'Drug' في شيت Drugs. الأعمدة: {drugs_df.columns.tolist()}")
            return None, None, None

        # شيت الكلمات المفتاحية (نبدأ من الصف التاني)
        keywords_df = pd.read_excel(excel_path, sheet_name="Keywords", header=1)
        keywords_df.columns = keywords_df.columns.str.strip().str.lower()
        keywords_df = keywords_df.loc[:, ~keywords_df.columns.str.contains('^unnamed', case=False)]  # تنظيف Unnamed

        # التأكد إن الأعمدة المطلوبة موجودة
        if not set(["keyword", "drug"]).issubset(keywords_df.columns):
            st.error(f"❌ الأعمدة المطلوبة مش موجودة في شيت Keywords. الأعمدة اللي لقيتها: {keywords_df.columns.tolist()}")
            return None, None, None

        # نسخة للعرض
        display_drugs_df = drugs_df.copy()

        # تجهيز للبحث (lowercase + strip)
        drugs_df["drug"] = drugs_df["drug"].astype(str).str.strip().str.lower()
        keywords_df["keyword"] = keywords_df["keyword"].astype(str).str.strip().str.lower()
        keywords_df["drug"] = keywords_df["drug"].astype(str).str.strip().str.lower()

        return drugs_df, keywords_df, display_drugs_df

    except Exception as e:
        st.error(f"❌ خطأ في قراءة ملف الإكسيل: {e}")
        return None, None, None


# ===== البحث في الأدوية أو الكلمات المفتاحية =====
def search_in_excel(query, drugs_df, keywords_df):
    query = query.lower().strip()

    # البحث في شيت Drugs → يرجّع كل الأعمدة الخاصة بالدواء
    if drugs_df is not None and "drug" in drugs_df.columns:
        match = drugs_df[drugs_df["drug"].str.contains(query, na=False)]
        if not match.empty:
            return "drug", match

    # البحث في شيت Keywords → يرجّع الأعمدة من شيت Keywords فقط (من غير keyword)
    if keywords_df is not None and {"keyword", "drug"}.issubset(keywords_df.columns):
        kw_match = keywords_df[keywords_df["keyword"].str.contains(query, na=False)]
        if not kw_match.empty:
            return "keyword", kw_match.drop(columns=["keyword"], errors="ignore")

    return None, None


# ===== تحميل وتقطيع ملفات PDF الطبية =====
def load_medical_docs(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# ===== تضمين النصوص =====
def embed_documents(docs):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(docs, embed_model)

# ===== حفظ واسترجاع قاعدة البيانات =====
def save_index(index):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

def load_index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return None

def update_index(new_docs):
    index = load_index()
    if index:
        index.add_documents(new_docs)
    else:
        index = embed_documents(new_docs)
    save_index(index)
    return index

# ===== إنشاء نظام الأسئلة والأجوبة =====
def build_qa_system(faiss_index):
    retriever = faiss_index.as_retriever(search_type="similarity", k=4)
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ===== استخراج النص من الصور =====
def extract_text_from_image(image):
    return pytesseract.image_to_string(image, lang="ara+eng")

# ===== الكشف عن اللغة =====
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# ===== تبسيط الاستعلام بناء على اللغة =====
def simplify_prompt(query, lang):
    if lang == "ar":
        return f"جاوب بالعربية العامية وبأسلوب سهل: {query}"
    elif lang == "en":
        return f"Answer in simple, conversational English: {query}"
    else:
        return query

# ===== دالة خاصة لفلترة إجابة الـ LLM =====
def ask_medical_qa(query):
    medical_prompt = f"""
    You are a helpful medical assistant.
    Answer ONLY about drugs, pharmacology, and medical knowledge.
    If the question is unrelated or you don't find info, reply exactly with:
    '⚠️ لم أجد معلومات طبية عن هذا الموضوع.'

    Question: {query}
    """
    return st.session_state.qa_chain.run(medical_prompt)

# ===== التطبيق الرئيسي =====
def main():
    st.set_page_config(page_title="CHATYMEDx", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #cba37d;'>CHATYMEDx</h1>", unsafe_allow_html=True)

    # تحميل أو إنشاء قاعدة البيانات
    if "qa_chain" not in st.session_state:
        if os.path.exists(INDEX_PATH):
            index = load_index()
        else:
            with st.spinner("📚 Embedding default medical file..."):
                docs = load_medical_docs("Clinical Pharmacology - D R Laurence.pdf")
                index = embed_documents(docs)
                save_index(index)
        st.session_state.qa_chain = build_qa_system(index)

    # تحميل بيانات الأدوية من Excel
    drugs_df, keywords_df, display_drugs_df = load_drugs_data(EXCEL_PATH)

    # ===== الشريط الجانبي =====
    with st.sidebar:
        if os.path.exists("Chaty_medx.jpg"):
        st.image("Chaty_medx.jpg", width=100)
        st.markdown("### 📘 Upload your PDF")
        pdf_file = st.file_uploader("PDF File", type=["pdf"])
        st.markdown("### 🖼️ Upload your image:")
        image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

    # ===== التعامل مع ملف PDF =====
    if pdf_file:
        with st.spinner("📄 Loading and embedding the file..."):
            with open("temp_medical.pdf", "wb") as f:
                f.write(pdf_file.read())
            docs = load_medical_docs("temp_medical.pdf")
            index = update_index(docs)
            st.session_state.qa_chain = build_qa_system(index)
            st.success("✅ File embedded and added to memory.")

    # ===== التعامل مع صورة =====
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="The uploaded image", use_container_width=True)
        with st.spinner("🧠 Extracting text from image..."):
            extracted_text = extract_text_from_image(image)
            st.text_area("Extracted text from image:", value=extracted_text, height=150)

    # ===== إدخال اسم الدواء أو كلمة مفتاحية =====
    query = st.text_input("Write drug name :")

    if query:
        kind, result_df = search_in_excel(query, drugs_df, keywords_df)

        if kind and result_df is not None and not result_df.empty:
            if kind == "drug":
                st.success(f"✅ Found drug: {query}")
            elif kind == "keyword":
                st.success(f"✅ Found related drug(s) for your keyword")
            st.dataframe(result_df)  # يعرض بيانات الدواء أو الأدوية بدون keyword وبدون Unnamed
        else:
            # fallback للـ QA system
            lang = detect_language(query)
            simplified_query = simplify_prompt(query, lang)
            result = ask_medical_qa(simplified_query)

            if lang == "ar":
                st.markdown(f"<div dir='rtl' style='text-align: right; font-size: 18px;'>{result}</div>", unsafe_allow_html=True)
                st.markdown("<div dir='rtl' style='text-align: right; font-size: 14px; color:gray;'>ملاحظة: هذه الخدمة لا تُعتبر بديلاً عن الاستشارة الطبية.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"### Your answer: {result}")
                st.markdown("Note: This service is not a substitute for professional medical advice.")

if __name__ == "__main__":
    main()







