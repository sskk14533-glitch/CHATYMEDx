import os
import streamlit as st
import pickle
from PIL import Image
from langdetect import detect
import pandas as pd
from docx import Document
import tempfile
import io

# ===== التحقق من وجود المكتبات الأساسية =====
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    st.warning("⚠️ Tesseract غير متوفر - خاصية استخراج النص من الصور معطلة")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# ===== التحقق من وجود المكتبات الثقيلة للـ AI =====
try:
    import torch
    import faiss
    TORCH_FAISS_AVAILABLE = True
except ImportError:
    TORCH_FAISS_AVAILABLE = False
    st.info("ℹ️ Torch أو FAISS غير موجودين - نظام AI لن يعمل الآن")

# ===== التحقق من LangChain (مشروط) =====
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("⚠️ مكتبات LangChain غير متوفرة - معالجة PDF ونظام AI لن يعمل")

INDEX_PATH = "faiss_index.pkl"
EXCEL_PATH = "Book3.xlsx"

# ===== تحميل بيانات الأدوية والكلمات المفتاحية =====
def load_drugs_data(excel_path):
    try:
        if not os.path.exists(excel_path):
            st.warning(f"⚠️ ملف الإكسيل {excel_path} غير موجود")
            return None, None, None

        drugs_df = pd.read_excel(excel_path, sheet_name="Drugs", header=0)
        drugs_df.columns = drugs_df.columns.str.strip().str.lower()
        drugs_df = drugs_df.loc[:, ~drugs_df.columns.str.contains('^unnamed', case=False)]

        if "drug" not in drugs_df.columns:
            st.error(f"❌ مفيش عمود 'Drug' في شيت Drugs. الأعمدة: {drugs_df.columns.tolist()}")
            return None, None, None

        keywords_df = pd.read_excel(excel_path, sheet_name="Keywords", header=1)
        keywords_df.columns = keywords_df.columns.str.strip().str.lower()
        keywords_df = keywords_df.loc[:, ~keywords_df.columns.str.contains('^unnamed', case=False)]

        if not set(["keyword", "drug"]).issubset(keywords_df.columns):
            st.error(f"❌ الأعمدة المطلوبة مش موجودة في شيت Keywords. الأعمدة اللي لقيتها: {keywords_df.columns.tolist()}")
            return None, None, None

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
    if keywords_df is not None and {"keyword", "drug"}.issubset(keywords_df.columns):
        kw_match = keywords_df[keywords_df["keyword"].str.contains(query, na=False)]
        if not kw_match.empty:
            return "keyword", kw_match.drop(columns=["keyword"], errors="ignore")
    return None, None

# ===== استخراج النص من الصور =====
def extract_text_from_image(image):
    if not TESSERACT_AVAILABLE:
        return "⚠️ خاصية استخراج النص من الصور غير متوفرة"
    try:
        return pytesseract.image_to_string(image, lang="ara+eng")
    except Exception as e:
        st.error(f"❌ خطأ في استخراج النص: {e}")
        return "فشل في استخراج النص"

# ===== الكشف عن اللغة =====
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def simplify_prompt(query, lang):
    if lang == "ar":
        return f"جاوب بالعربية العامية وبأسلوب سهل: {query}"
    elif lang == "en":
        return f"Answer in simple, conversational English: {query}"
    else:
        return query

# ===== حفظ واسترجاع فهرس AI =====
def save_index(index):
    try:
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(index, f)
    except Exception as e:
        st.error(f"❌ خطأ في حفظ الفهرس: {e}")

def load_index():
    try:
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الفهرس: {e}")
    return None

def embed_documents(docs):
    if not TORCH_FAISS_AVAILABLE or not LANGCHAIN_AVAILABLE:
        return None
    try:
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_documents(docs, embed_model)
    except Exception as e:
        st.error(f"❌ خطأ في التضمين: {e}")
        return None

def build_qa_system(faiss_index):
    if not TORCH_FAISS_AVAILABLE or not LANGCHAIN_AVAILABLE:
        return None
    try:
        retriever = faiss_index.as_retriever(search_type="similarity", k=4)
        if "GROQ_API_KEY" not in st.secrets:
            st.error("❌ GROQ_API_KEY مش موجود في secrets")
            return None
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(api_key=api_key, model_name="llama3-8b-8192")
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        st.error(f"❌ خطأ في إنشاء نظام QA: {e}")
        return None

# ===== التطبيق الرئيسي =====
def main():
    st.set_page_config(page_title="CHATYMEDx", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #cba37d;'>CHATYMEDx</h1>", unsafe_allow_html=True)

    # تحميل بيانات Excel
    drugs_df, keywords_df, display_drugs_df = load_drugs_data(EXCEL_PATH)

    # الشريط الجانبي
    with st.sidebar:
        logo_path = "Chaty_medx.jpg"
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        st.markdown("### 📘 Upload your PDF")
        pdf_file = st.file_uploader("PDF File", type=["pdf"])
        st.markdown("### 🖼️ Upload your image")
        image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

    # معالجة رفع الصور
    if image_file:
        try:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded image", use_container_width=True)
            extracted_text = extract_text_from_image(image)
            st.text_area("Extracted text:", value=extracted_text, height=150)
        except Exception as e:
            st.error(f"❌ خطأ في معالجة الصورة: {e}")

    # مربع البحث
    query = st.text_input("Write drug name:")
    if query:
        kind, result_df = search_in_excel(query, drugs_df, keywords_df)
        if kind and result_df is not None and not result_df.empty:
            if kind == "drug":
                st.success(f"✅ Found drug: {query}")
            elif kind == "keyword":
                st.success(f"✅ Found related drug(s) for your keyword")
            st.dataframe(result_df)
        else:
            if TORCH_FAISS_AVAILABLE and LANGCHAIN_AVAILABLE and os.path.exists(INDEX_PATH):
                index = load_index()
                qa_chain = build_qa_system(index)
                st.session_state.qa_chain = qa_chain
                if qa_chain:
                    lang = detect_language(query)
                    simplified_query = simplify_prompt(query, lang)
                    result = qa_chain.run(simplified_query)
                    st.markdown(f"### Your answer: {result}")

    # معلومات النظام
    with st.expander("ℹ️ System Status"):
        st.write(f"📷 Tesseract Available: {TESSERACT_AVAILABLE}")
        st.write(f"🔊 gTTS Available: {GTTS_AVAILABLE}")
        st.write(f"🗃️ Excel Data Available: {drugs_df is not None}")
        st.write(f"🤖 QA System Ready: {TORCH_FAISS_AVAILABLE and LANGCHAIN_AVAILABLE}")

if __name__ == "__main__":
    main()



