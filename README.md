# 🧠 RAG-Based Question Answering Assistant using LLMs

## 🤖 What is this?

A **Retrieval-Augmented Generation (RAG)** powered assistant built using **LangChain** and **vector databases (FAISS/Chroma)** that can answer user questions based on custom documents.  
This project was developed as part of **AAIDC Module 1 – “Foundations of Agentic AI: Your First RAG Assistant”**.

**Think of it as:** Your PA that knows about YOUR documents and can answer questions about them.

---

## 🚀 Project Overview

This project demonstrates how **Large Language Models (LLMs)** can be enhanced using **retrieval-based augmentation** to provide accurate, context-specific answers.  
Instead of relying solely on the model’s internal knowledge, it retrieves relevant information from ingested documents, enabling domain-specific Q&A.

You can interact with this assistant through a **Command-Line Interface (CLI)**, and optionally via a **Streamlit UI** for an improved user experience.

---

## 🧩 Key Features

- 📄 **Custom Document Ingestion** – Upload and index your own documents (PDFs, text, or markdown).  
- 🔍 **Vector Store Retrieval** – Uses FAISS or Chroma to store embeddings and retrieve relevant context.  
- 🤖 **LLM-Powered Responses** – Generates natural, context-aware answers using OpenAI, Groq, or Gemini APIs.  
- 🧠 **LangChain Integration** – Handles the end-to-end prompt → retrieval → response pipeline.  
- 🧾 **Configurable Pipeline** – Supports easy switching between vector databases and models.  
- 🗂️ **Extensible Design** – Can be enhanced with session memory, logging, or reasoning chains (ReAct, CoT).  

---

## 🧱 Tech Stack

| Component | Technology |
|------------|-------------|
| Framework | **LangChain** |
| Vector Store | **FAISS** or **Chroma** |
| Embeddings | **OpenAI**, **Groq**, or **Gemini** |
| LLM Backend | **OpenAI GPT**, **Gemini**, or other API models |
| Interface | **CLI** (with optional **Streamlit** UI) |
| Language | **Python 3.10+** |

---

## 📂 Folder Structure

```
rt-aaidc-module1/
├── src/
│   ├── app.py           # Main RAG application
│   └── vectordb.py      # Vector database wrapper
├── data/               # Contains documnets
│   ├── *.txt          # Contains text files
├── requirements.txt    # All dependencies included
└── README.md          # This guide
```




## 🎓 Learning Outcomes

By completing this project, you’ll learn to:
- Implement a **Retrieval-Augmented Generation (RAG)** pipeline  
- Use **vector databases** for semantic search  
- Work with **embeddings** and **LLMs**  
- Design **prompt templates** for context-driven answers  
- Build real-world **AI assistant applications**

---






## ⚙️ Installation & Setup

Follow the steps below to set up and run the project locally:

### 1️⃣ Clone the repository
   ```bash
   git clone https://github.com/krishpansara/rt-aaidc-module1/
   cd rt-aaidc-module1
   ```

### 2️⃣ Create and Activate a Virtual Environment

It’s recommended to create a virtual environment to isolate project dependencies.

#### 🪟 For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### 🐧 For macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install the Required Dependencies

Once the virtual environment is activated, install all dependencies using:
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure your API key:

This project supports multiple LLM providers (OpenAI, Groq, Google).  
You need to set your API keys before running the app.

After creating the Virtual Environment your project containes `.env` file in the project root directory add your API keys in the file as shown below:

```bash
# .env file

# Example: use one or more providers depending on your setup
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 5️⃣ Run the Application
```bash
  python app.py
```

---


## 👤 Author

**Krish Pansara**  
💻 Passionate about AI, ML, and building intelligent applications.  
🔗 [GitHub Profile](https://github.com/krishpansara)

