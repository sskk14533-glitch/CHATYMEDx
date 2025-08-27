
import os
import streamlit as st
from PIL import Image
import pandas as pd

EXCEL_PATH = "Book3 (2).xlsx"

# ===== تحميل بيانات الأدوية من Excel =====
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



# ===== التطبيق الرئيسي =====
def main():
    st.set_page_config(page_title="CHATYMEDx", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #cba37d;'>CHATYMEDx</h1>", unsafe_allow_html=True)

    # تحميل بيانات الأدوية
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

    

    # التعامل مع الصور
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="The uploaded image", use_container_width=True)

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
                st.warning("⚠️ لم يتم العثور على الدواء أو الكلمة المفتاحية.")

if __name__ == "__main__":
    main()

