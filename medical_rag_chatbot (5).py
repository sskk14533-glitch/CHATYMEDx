
import os
import streamlit as st
from PIL import Image
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS
import requests
from typing import Optional, List, Any

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

# ===== Groq LLM مخصص (معطل البحث المتقدم) =====
class GroqLLM:
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model_name = model_name

    def run(self, prompt: str) -> str:
        return "⚠️ ميزة البحث المتقدم مع Groq غير مفعلة في النسخة الحالية."

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
        with st.spinner("📄 Loading the file..."):
            with open("temp_medical.pdf", "wb") as f:
                f.write(pdf_file.read())
            docs = load_medical_docs("temp_medical.pdf")
            st.success(f"✅ PDF loaded successfully ({len(docs)} pages/chunks).")

    # التعامل مع صورة
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
                st.warning("⚠️ لم يتم العثور على معلومات في Excel. ميزة البحث المتقدم غير متاحة.")

if __name__ == "__main__":
    main()



