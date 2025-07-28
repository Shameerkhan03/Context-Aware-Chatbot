# Mistral RAG Chatbot 🧠💬

A Retrieval-Augmented Generation (RAG) based chatbot that answers questions from your own documents using **LangChain**, **FAISS**, and **Mistral** LLM via **Ollama**. Built with a simple and clean **Streamlit UI**.

---

## 🚀 Features

- 🔎 Ask questions grounded in your document context
- 🧠 Local LLM support (via Ollama with Mistral)
- 🗃️ Vector similarity search using FAISS or ChromaDB
- 💡 LangChain integration for prompt chaining
- 🌐 Minimal, responsive Streamlit web interface

---

## 📸 UI Preview

<img src="screenshots\dashboard.png" alt="Chatbot Dashboard" width="600"/>
<br><br>
<img src="screenshots\sample queries.png" alt="Sample Queries" width="600"/>

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/context-aware-chatbot.git
cd context-aware-chatbot
```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies


```bash
pip install -r requirements.txt
```

### 4. Run Ollama and Pull the Model


| ⚠️ Ensure Ollama is installed and running on your system.

```bash
ollama pull mistral
```

### 5. Launch the App


```bash
streamlit run app.py

```