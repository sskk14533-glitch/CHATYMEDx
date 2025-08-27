
import os
import streamlit as st
import pickle
from PIL import Image
import pandas as pd

# ===== التأكد من مكتبة openpyxl =====
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    st.warning("⚠️ مكتبة openpyxl غير موجودة. قراءة ملفات Excel ستكون معطلة.")

# ===== تحميل بيانات الأدوية والكلمات المفتاحية =====
EXCEL_PATH = "Book3.xlsx"

def load_drugs_data(excel_path):
    if not OPENPYXL_AVAILABLE:
        st.error("❌ لا يمكن قراءة ملف الإكسيل بدون مكتبة openpyxl")
        return None, None, None

    try:
        if not os.path.exists(excel_path):
            st.warning(f"⚠️ ملف الإكسيل {excel_path} غير موجود")
            return None, None, None

        drugs_df = pd.read_excel(excel_path, sheet_name="Drugs", header=0)
        drugs_df.columns = drugs_df.columns.str.strip().str.lower()
        keywords_df = pd.read_excel(excel_path, sheet_name="Keywords", header=1)
        keywords_df.columns = keywords_df.columns.str.strip().str.lower()
        return drugs_df, keywords_df, drugs_df.copy()
    except Exception as e:
        st.error(f"❌ خطأ في قراءة ملف الإكسيل: {e}")
        return None, None, None

def search_in_excel(query, drugs_df, keywords_df):
    query = query.lower().strip()
    if drugs_df is not None:
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

    drugs_df, keywords_df, _ = load_drugs_data(EXCEL_PATH)

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
            st.warning("⚠️ لم يتم العثور على الدواء أو الكلمة المفتاحية.")

if __name__ == "__main__":
    main()






