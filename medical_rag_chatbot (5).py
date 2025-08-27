import os
import streamlit as st
import pickle
from PIL import Image
from langdetect import detect
import pandas as pd
from docx import Document
import tempfile
import io

# استيراد المكتبات مع معالجة الأخطاء
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    st.warning("⚠️ Tesseract غير متوفر - خاصية استخراج النص من الصور معطلة")

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
    st.error("❌ مكتبات LangChain غير متوفرة")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

INDEX_PATH = "faiss_index.pkl"
EXCEL_PATH = "Book3.xlsx"

# ===== تحميل بيانات الأدوية والكلمات المفتاحية من Excel =====
def load_drugs_data(excel_path):
    try:
        # التحقق من وجود الملف
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


# ===== البحث في الأدوية أو الكلمات المفتاحية =====
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


# ===== تحميل وتقطيع ملفات PDF الطبية =====
def load_medical_docs(file_path):
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain غير متوفر لمعالجة PDF")
        return []
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)
    except Exception as e:
        st.error(f"❌ خطأ في تحميل PDF: {e}")
        return []


# ===== تضمين النصوص =====
def embed_documents(docs):
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain غير متوفر للتضمين")
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


# ===== حفظ واسترجاع قاعدة البيانات =====
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


def update_index(new_docs):
    if not new_docs:
        return None
        
    try:
        index = load_index()
        if index:
            index.add_documents(new_docs)
        else:
            index = embed_documents(new_docs)
        if index:
            save_index(index)
        return index
    except Exception as e:
        st.error(f"❌ خطأ في تحديث الفهرس: {e}")
        return None


# ===== إنشاء نظام الأسئلة والأجوبة =====
def build_qa_system(faiss_index):
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain غير متوفر لنظام QA")
        return None
        
    try:
        retriever = faiss_index.as_retriever(search_type="similarity", k=4)
        
        # التحقق من وجود API key
        if "GROQ_API_KEY" not in st.secrets:
            st.error("❌ GROQ_API_KEY مش موجود في secrets")
            return None
            
        api_key = st.secrets["GROQ_API_KEY"]

        llm = ChatGroq(
            api_key=api_key,
            model_name="llama3-8b-8192"
        )

        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        st.error(f"❌ خطأ في إنشاء نظام QA: {e}")
        return None


# ===== استخراج النص من الصور =====
def extract_text_from_image(image):
    if not TESSERACT_AVAILABLE:
        st.warning("⚠️ Tesseract غير متوفر لاستخراج النص من الصور")
        return "خاصية استخراج النص من الصور غير متوفرة حاليا"
    
    try:
        return pytesseract.image_to_string(image, lang="ara+eng")
    except Exception as e:
        st.error(f"❌ خطأ في استخراج النص: {e}")
        return "فشل في استخراج النص من الصورة"


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
    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
        return "⚠️ نظام الأسئلة والأجوبة غير متوفر حاليا"
        
    try:
        medical_prompt = f"""
        You are a helpful medical assistant.
        Answer ONLY about drugs, pharmacology, and medical knowledge.
        If the question is unrelated or you don't find info, reply exactly with:
        '⚠️ لم أجد معلومات طبية عن هذا الموضوع.'

        Question: {query}
        """
        return st.session_state.qa_chain.run(medical_prompt)
    except Exception as e:
        st.error(f"❌ خطأ في الاستعلام: {e}")
        return "⚠️ حدث خطأ أثناء البحث"


# ===== التطبيق الرئيسي =====
def main():
    st.set_page_config(page_title="CHATYMEDx", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #cba37d;'>CHATYMEDx</h1>", unsafe_allow_html=True)

    # إنشاء نظام QA إذا لم يكن موجود
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
        
        if LANGCHAIN_AVAILABLE:
            # التحقق من وجود ملف PDF الافتراضي
            default_pdf = "Clinical Pharmacology - D R Laurence.pdf"
            if os.path.exists(INDEX_PATH):
                index = load_index()
                if index:
                    st.session_state.qa_chain = build_qa_system(index)
            elif os.path.exists(default_pdf):
                with st.spinner("📚 Embedding default medical file..."):
                    docs = load_medical_docs(default_pdf)
                    if docs:
                        index = embed_documents(docs)
                        if index:
                            save_index(index)
                            st.session_state.qa_chain = build_qa_system(index)

    # تحميل بيانات الأدوية
    drugs_df, keywords_df, display_drugs_df = load_drugs_data(EXCEL_PATH)

    # الشريط الجانبي
    with st.sidebar:
        # عرض الصورة إذا كانت موجودة
        logo_path = "Chaty_medx.jpg"
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.info("💡 يمكنك إضافة logo بالاسم 'Chaty_medx.jpg'")

        st.markdown("### 📘 Upload your PDF")
        pdf_file = st.file_uploader("PDF File", type=["pdf"])

        st.markdown("### 🖼️ Upload your image:")
        image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

    # معالجة رفع PDF
    if pdf_file and LANGCHAIN_AVAILABLE:
        with st.spinner("📄 Loading and embedding the file..."):
            try:
                # استخدام tempfile لإنشاء ملف مؤقت آمن
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_path = tmp_file.name
                
                docs = load_medical_docs(tmp_path)
                if docs:
                    index = update_index(docs)
                    if index:
                        st.session_state.qa_chain = build_qa_system(index)
                        st.success("✅ File embedded and added to memory.")
                    else:
                        st.error("❌ فشل في إنشاء الفهرس")
                else:
                    st.error("❌ فشل في تحميل المستندات")
                
                # حذف الملف المؤقت
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ خطأ في معالجة PDF: {e}")

    # معالجة رفع الصور
    if image_file:
        try:
            image = Image.open(image_file)
            st.image(image, caption="The uploaded image", use_container_width=True)
            
            if TESSERACT_AVAILABLE:
                with st.spinner("🧠 Extracting text from image..."):
                    extracted_text = extract_text_from_image(image)
                    st.text_area("Extracted text from image:", value=extracted_text, height=150)
            else:
                st.warning("⚠️ خاصية استخراج النص من الصور غير متوفرة")
        except Exception as e:
            st.error(f"❌ خطأ في معالجة الصورة: {e}")

    # مربع البحث الرئيسي
    query = st.text_input("Write drug name:")

    if query:
        # البحث في بيانات الأدوية أولا
        kind, result_df = search_in_excel(query, drugs_df, keywords_df)

        if kind and result_df is not None and not result_df.empty:
            if kind == "drug":
                st.success(f"✅ Found drug: {query}")
            elif kind == "keyword":
                st.success(f"✅ Found related drug(s) for your keyword")
            st.dataframe(result_df)
        else:
            # إذا لم نجد في الأدوية، نبحث باستخدام AI
            if LANGCHAIN_AVAILABLE and st.session_state.qa_chain:
                lang = detect_language(query)
                simplified_query = simplify_prompt(query, lang)
                result = ask_medical_qa(simplified_query)

                if lang == "ar":
                    st.markdown(f"<div dir='rtl' style='text-align: right; font-size: 18px;'>{result}</div>", unsafe_allow_html=True)
                    st.markdown("<div dir='rtl' style='text-align: right; font-size: 14px; color:gray;'>ملاحظة: هذه الخدمة لا تُعتبر بديلاً عن الاستشارة الطبية.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### Your answer: {result}")
                    st.markdown("Note: This service is not a substitute for professional medical advice.")
            else:
                st.warning("⚠️ نظام الذكي الاصطناعي غير متوفر حاليا")

    # إضافة معلومات عن حالة النظام
    with st.expander("ℹ️ System Status"):
        st.write(f"🔧 LangChain Available: {LANGCHAIN_AVAILABLE}")
        st.write(f"📷 Tesseract Available: {TESSERACT_AVAILABLE}")
        st.write(f"🔊 gTTS Available: {GTTS_AVAILABLE}")
        st.write(f"🗃️ Excel Data Available: {drugs_df is not None}")
        st.write(f"🤖 QA System Ready: {st.session_state.qa_chain is not None}")


if __name__ == "__main__":
    main()



