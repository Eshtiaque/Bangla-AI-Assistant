# Bangla AI Chatbot (RAG + Voice Enabled)

This is a **Retrieval-Augmented Generation (RAG)** based AI Chatbot capable of answering questions in Bengali on specific topics. It uses **Groq (Llama 3.1)** for reasoning and **gTTS** for voice output.

## 🚀 Features
- **Topic Detection:** Automatically detects topics (Education, Health, Sports, Technology, Travel).
- **RAG Architecture:** Fetches answers strictly from a predefined local dataset (`dataset.py`) to prevent hallucinations.
- **Voice Output (TTS):** Reads the answer aloud in Bengali using Google Text-to-Speech.
- **Demo Questions:** Sidebar with clickable/viewable demo questions for testing.
- **Fast Response:** Powered by the `llama-3.1-8b-instant` model via Groq API.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM:** Groq (Llama 3.1)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector DB:** ChromaDB
- **Audio:** gTTS (Google Text-to-Speech)

---

## ⚙️ Installation & Setup

Follow these steps to run the project on your local machine.

### 1. Prerequisites
- Python 3.8 or higher installed.
- A Groq API Key (Get it from [console.groq.com](https://console.groq.com)).

### 2. Clone or Download
Download the project folder and navigate to it in your terminal.

```bash
cd Bangla_Chatbot
```

# 3. Create a Virtual Environment (Optional but Recommended)
Bash

### Windows
```
python -m venv venv
```
```
.\venv\Scripts\activate
```
### Mac/Linux
```
python3 -m venv venv
```
```
source venv/bin/activate
```
# 4. Install Dependencies
Make sure you have requirements.txt in the folder. Then run:

```
pip install -r requirements.txt
```

# 5. Set API Key
You can set the API key in two ways:

Create a .env file in the root folder and add:
```
GROQ_API_KEY=your_actual_api_key_here
OR, open app.py and manually paste your key in the api_key variable fallback section.
```


# ▶️ How to Run
Run the Streamlit app using the following command:

```
streamlit run app.py
```
The app will open automatically in your browser at http://localhost:8501.
``` 

📂 Project Structure
Bangla_Chatbot/
│

├── app.py   
├── dataset.py         
├── requirements.txt   
├── .env                
└── README.md           

You can ask questions related to:
```
Education (শিক্ষা) - e.g.,
 "এইচএসসি পরীক্ষা কবে হতে পারে?"

Health (স্বাস্থ্য) - e.g., "ডেঙ্গু জ্বরের লক্ষণ কী?"

Sports (খেলাধুলা) - e.g., "লিওনেল মেসি কোন দেশের খেলোয়াড়?"

Technology (প্রযুক্তি) - e.g., "র‍্যাম এর কাজ কী?"

Travel (ভ্রমণ) - e.g., "সাজেক ভ্যালি কোথায় অবস্থিত?"
```

# ⚠️ Troubleshooting
Error: streamlit is not recognized: Make sure you installed the requirements and activated the virtual environment.

Error: API Key missing: Ensure your Groq API key is valid and placed correctly in .env or app.py.

Audio not playing: Ensure you have a stable internet connection for gTTS to generate audio.