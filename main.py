__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from gtts import gTTS
from io import BytesIO
from dotenv import load_dotenv

# --- Imports (Updated for Latest LangChain v0.3) ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 🔥 Update 1: PromptTemplate in langchain_core 
from langchain_core.prompts import PromptTemplate 

# 🔥 Update 2: RetrievalQA 
from langchain.chains import RetrievalQA

# Import data from dataset.py
from dataset import get_data 

# --- Page Configuration ---
st.set_page_config(page_title="Bangla AI Chatbot", page_icon="🤖", layout="wide")

# --- 1. Setup API Key ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Fallback if .env is missing
if not api_key:
    # 👇 Paste your real API Key here
    api_key = "gsk_TOMAR_REAL_API_KEY_BOSHAO"

# --- 2. Initialize System (Cached for performance) ---
@st.cache_resource
def initialize_chatbot():
    # Load data from dataset.py
    docs = get_data()
    
    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    # Initialize LLM (Llama 3.1 8b Instant - Fastest)
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.1-8b-instant", 
        temperature=0
    )
    return vectorstore, llm

# Attempt to initialize the system
try:
    vectorstore, llm = initialize_chatbot()
except Exception as e:
    st.error(f"❌ Setup Error: {e}")
    st.stop()

# --- 3. Sidebar: Demo Questions ---
with st.sidebar:
    st.title("📌 Demo Questions")
    st.info("Try asking these questions to test the bot:")
    
    st.markdown("""
    **1. Education (শিক্ষা):**
    * এইচএসসি পরীক্ষা কবে হতে পারে?
    
    **2. Health (স্বাস্থ্য):**
    * ডেঙ্গু জ্বরের লক্ষণ কী?
    
    **3. Sports (খেলাধুলা):**
    * লিওনেল মেসি কোন দেশের খেলোয়াড়?
    
    **4. Technology (প্রযুক্তি):**
    * র‍্যাম (RAM) এর কাজ কী?
    
    **5. Travel (ভ্রমণ):**
    * সাজেক ভ্যালি কোথায় অবস্থিত?
    """)
    st.write("---")
    st.caption("© Bangla RAG Chatbot Project")

# --- 4. Helper Function: Topic Detection ---
def detect_topic(query, llm):
    """
    Detects the topic of the user's query and maps it to the 
    dataset's metadata keys (e.g., 'shiksha', 'projukti').
    """
    q_lower = query.lower()

    # Strategy 1: Manual Keyword Checking (High Accuracy)
    if any(x in q_lower for x in ['এইচএসসি', 'বিশ্ববিদ্যালয়', 'পড়া', 'exam', 'hsc', 'cse']): return 'shiksha'
    if any(x in q_lower for x in ['জ্বর', 'ডেঙ্গু', 'চিকিৎসা', 'ঔষধ', 'health', 'গ্যাস্ট্রিক']): return 'shastho'
    if any(x in q_lower for x in ['খেলা', 'ক্রিকেট', 'ফুটবল', 'মেসি', 'sports', 'ক্যাপ্টেন']): return 'kheladhula'
    if any(x in q_lower for x in ['এআই', 'ai', 'ram', 'python', 'computer', 'র‍্যাম']): return 'projukti'
    if any(x in q_lower for x in ['কক্সবাজার', 'সাজেক', 'ভ্রমণ', 'tour', 'visa']): return 'vromon'

    # Strategy 2: AI Fallback (If keywords miss)
    valid_topics = ['education', 'health', 'sports', 'technology', 'travel']
    prompt = f"Classify topic: {valid_topics}. Return ONLY the topic name. Question: {query}"
    
    try:
        response = llm.invoke(prompt)
        ai_topic = response.content.strip().lower()
        
        mapping = {
            "education": "shiksha", "health": "shastho", "sports": "kheladhula",
            "technology": "projukti", "travel": "vromon"
        }
        
        for key in mapping:
            if key in ai_topic: return mapping[key]
        return "unknown"
    except:
        return "unknown"

# --- 5. Helper Function: Text to Speech ---
def text_to_speech(text):
    """
    Generate Audio from Text using Google TTS
    """
    try:
        tts = gTTS(text=text, lang='bn')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        return audio_fp
    except:
        return None

# --- 6. Main UI Layout ---
st.title("🤖 Bangla AI Assistant (Voice Enabled 🔊)")
st.markdown("Ask questions about: **Education, Health, Sports, Technology, Travel**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. Chat Logic (RAG) ---
if query := st.chat_input("আপনার প্রশ্ন লিখুন... (Type your question)"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # A. Detect Topic
        topic = detect_topic(query, llm)
        
        # Mapping for UI Display
        display_map = {
            "shiksha": "শিক্ষা (Education)", "shastho": "স্বাস্থ্য (Health)", 
            "kheladhula": "খেলাধুলা (Sports)", "projukti": "প্রযুক্তি (Technology)", 
            "vromon": "ভ্রমণ (Travel)", "unknown": "অজানা (Unknown)"
        }
        display_text = display_map.get(topic, "Unknown")
        
        # Variable to store text for voice generation
        voice_text = "" 

        # B. Handle Unknown Topics
        if topic == "unknown":
            response_text = "দুঃখিত, আমি এই বিষয়ে জানি না। দয়া করে বামপাশের ডেমো প্রশ্নগুলো চেষ্টা করুন।"
            voice_text = response_text
        
        else:
            try:
                # C. Strict Prompt (Prevents Hallucinations)
                template = """
                You are a helpful assistant. Answer the question based ONLY on the provided Context.
                
                Context: {context}
                Question: {question}
                
                Rules:
                1. If the answer is in the context, output it exactly.
                2. If the answer is NOT in the context, say "দুঃখিত, আমার ডাটাবেজে সঠিক উত্তর নেই।".
                3. Do not make up any information.
                
                Answer in Bangla:
                """
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

                # D. Retrieve Documents (Filtered by Topic)
                retriever = vectorstore.as_retriever(
                    search_kwargs={"filter": {"topic": topic}, "k": 1}
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                )
                
                # E. Generate Answer
                res = qa_chain.invoke(query)
                answer = res['result']

                clean_answer = answer.replace("উত্তর:", "").replace("Answer:", "").strip()
                
                response_text = f"**বিষয়:** {display_text}\n\n{clean_answer}"
                voice_text = clean_answer # Only speak the answer part
                
            except Exception as e:
                response_text = f"Error: {e}"
                voice_text = "দুঃখিত, একটি টেকনিক্যাল সমস্যা হয়েছে।"

        # Display assistant response
        message_placeholder.markdown(response_text)
        
        # --- 🔊 Play Audio (TTS Feature) ---
        audio_data = text_to_speech(voice_text)
        if audio_data:
            st.audio(audio_data, format="audio/mp3")
        # -----------------------------------

        st.session_state.messages.append({"role": "assistant", "content": response_text})