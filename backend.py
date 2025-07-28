import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# --- Config ---
BLOG_DIR = "blogs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "faiss_index"
LLM_MODEL_NAME = "mistral"

qa_chain = None


# --- Step 1: Data Loading and Chunking ---
def load_and_chunk_documents(blog_dir=BLOG_DIR, chunk_size=200, chunk_overlap=20):
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    for filename in os.listdir(blog_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(blog_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n", 1)
                title = lines[0].replace("Title: ", "").strip() if lines else "No Title"
                text_content = lines[1].strip() if len(lines) > 1 else content.strip()
                chunks = text_splitter.create_documents([text_content])
                for i, chunk in enumerate(chunks):
                    chunk.metadata = {"title": title, "chunk_index": i}
                    docs.append(chunk)
    return docs


# --- Step 2: Vector Store ---
def create_and_save_vector_store(
    documents, model_name=EMBEDDING_MODEL_NAME, path=VECTOR_STORE_PATH
):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(path)
    return vector_store


def load_vector_store(model_name=EMBEDDING_MODEL_NAME, path=VECTOR_STORE_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# --- Step 3: RAG Pipeline Initialization ---
def initialize_rag_pipeline():
    global qa_chain
    if qa_chain is not None:
        return qa_chain

    if not os.path.exists(VECTOR_STORE_PATH):
        docs = load_and_chunk_documents()
        faiss_vector_store = create_and_save_vector_store(docs)
    else:
        faiss_vector_store = load_vector_store()

    # Load Mistral via Ollama
    llm = OllamaLLM(model=LLM_MODEL_NAME)

    retriever = faiss_vector_store.as_retriever()

    prompt_template = """You are a helpful assistant that answers questions STRICTLY based on the provided context. If the answer is not explicitly found in the context, clearly state that you do not have enough information in the provided text to answer.

Context:
{context}

Question:
{question}

Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    return qa_chain


# --- Step 4: Querying Function ---
def get_answer(query, chat_history):
    global qa_chain
    if qa_chain is None:
        qa_chain = initialize_rag_pipeline()

    result = qa_chain.invoke({"question": query, "chat_history": chat_history})

    full_output = result["answer"]

    # Clean output
    answer = full_output.split("Answer:", 1)[-1].strip()
    answer = answer.split("Question:")[0].split("Context:")[0].strip()

    return answer
