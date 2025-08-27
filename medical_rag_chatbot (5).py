
import os
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
from gtts import gTTS
import PyPDF2
from typing import Optional

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

# ===== استخراج نص من PDF =====
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"❌ خطأ في قراءة ملف PDF: {e}")
    return text

# ===== تحويل النص إلى صوت =====
def text_to_speech(text, filename="output.mp3"):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"❌ خطأ في تحويل النص إلى صوت: {e}")
        return None

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
    pdf_text = ""
    if pdf_file:
        with st.spinner("📄 Extracting text from PDF..."):
            with open("temp_medical.pdf", "wb") as f:
                f.write(pdf_file.read())
            pdf_text = extract_text_from_pdf("temp_medical.pdf")
            st.text_area("📄 PDF Text", pdf_text, height=300)

    # التعامل مع صورة
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="The uploaded image", use_container_width=True)

    # إدخال اسم الدواء أو كلمة مفتاحية
    query = st.text_input("Write drug name or keyword:")
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
                st.warning("⚠️ لم يتم العثور على نتيجة في Excel.")

    # تحويل النص الموجود في PDF إلى صوت
    if pdf_text:
        if st.button("🔊 Convert PDF text to speech"):
            audio_file = text_to_speech(pdf_text)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()



