
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        " GEMINI_API_KEY not found.\n"
        "Set it in terminal using:\n"
        "  PowerShell: $env:GEMINI_API_KEY='your_api_key'\n"
        "  Command Prompt: set GEMINI_API_KEY=your_api_key\n"
        "Or put it in a .env file like:\n"
        "  GEMINI_API_KEY=your_api_key"
    )

genai.configure(api_key=GEMINI_API_KEY)

vectorstore = None  

def load_file(file_path: str):
    """
    Load a PDF, DOCX, or TXT file and return documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        try:
            from langchain.document_loaders import UnstructuredWordDocumentLoader
        except ImportError:
            raise ImportError("Install 'unstructured' and 'python-docx' for DOCX support.")
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, TXT are allowed.")

    docs = loader.load()
    return docs


def initialize_vectorstore(file_path: str):
    """
    Load document, split into chunks, create embeddings, and initialize vectorstore.
    """
    global vectorstore
    docs = load_file(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = Chroma.from_documents(chunks, embeddings)


def rag_query(question: str) -> str:
    if vectorstore is None:
        raise ValueError("Vectorstore not initialized. Call initialize_vectorstore(file_path) first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


if __name__ == "__main__":
    file_path = input("Enter the path of your document (PDF, DOCX, TXT): ").strip()
    initialize_vectorstore(file_path)

    print("\nGemini RAG bot ready! Type 'exit' to quit.")
    while True:
        q = input("\nAsk: ")
        if q.lower() == "exit":
            print(" Goodbye!")
            break
        answer = rag_query(q)
        print("\n Answer:", answer)


