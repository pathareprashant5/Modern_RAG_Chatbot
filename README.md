# 💬 Modern RAG Chatbot

A sleek **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit** and **LangChain**, allowing you to ask questions and get answers directly sourced from websites you add. Answers include **clickable references** to the original sources for easy verification.

---

## 🔹 Features

- **Add websites dynamically**.  
- **Remove websites safely** without crashing the app.  
- Answers are generated **only from the content of the websites you added**.  
- **Clickable sources** in answers, opening in a new tab.  
- **Live streaming** of LLM responses using **Groq LLaMA-3.1**.  
- Optional **chat history display** with source links.  
- Copy answers directly for your convenience.  

---

## ⚡ How It Works

1. **Add Websites** – Enter the URL to add sources for the chatbot.  
2. **Document Loading** – The bot scrapes the web pages, splits them into chunks, and stores embeddings in a **Chroma vector store**.  
3. **Ask Questions** – Type your query; the bot retrieves relevant chunks and generates an answer.  
4. **Sources Extraction** – Only the chunks actually used in the answer are shown as clickable links.  
5. **Chat History** – Optionally view previous questions and answers with sources.  

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- **Streamlit** – for the interactive web UI  
- **LangChain** – for RAG pipeline  
- **Chroma** – vector database for document embeddings  
- **OpenAI Embeddings** – text embeddings  
- **Groq LLaMA-3.1-8B** – LLM for answer generation  

---

## 🚀 Getting Started

1️⃣ Clone the repository

```bash
git clone https://github.com/pathareprashant5/web-rag-chatbot.git
cd web-rag-chatbot

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Set up API keys
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key

4️⃣ Run the app
streamlit run rag.py

📌 Usage
    1. Add websites.

    2. Ask a question in the main panel.

    3. The bot will stream its answer, and below it will display sources that were actually used.

    4. Remove websites using the ❌ buttons in the sidebar if needed.

    5. Enable Show Chat History in the sidebar to review past interactions

🔗 Example
Question: What is LangChain?

Answer: LangChain is a framework for building applications powered by language models...
Sources:
- 🔗 [View Source](https://www.langchain.com/docs)

💡 Future Improvements

Support PDFs, YouTube transcripts, and local documents as additional sources.

Add user authentication to save private chat history.

Improve source highlighting in the answer text

📝 License

This project is licensed under MIT License – see the LICENSE file for details.