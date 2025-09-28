import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from sentence_transformers import SentenceTransformer
import ollama

# ---------------- Embedding ----------------
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="D:/huggingface_models"
)

class LocalEmbedding:
    def embed_documents(self, texts):
        return embed_model.encode(
            texts, show_progress_bar=False, convert_to_tensor=False
        ).tolist()
    
    def embed_query(self, text):
        return embed_model.encode(
            [text], convert_to_tensor=False
        ).tolist()[0]

# ---------------- Vectorstore ----------------
vectorstore = None
current_file_path = None
persist_folder = "D:/chroma_vectorstore"  # Folder to store vectorstores

def initialize_vectorstore(file_path: str):
    """
    Initialize or load a persistent Chroma vectorstore from PDF.
    """
    global vectorstore, current_file_path

    if vectorstore is not None and current_file_path == file_path:
        print("Using existing vectorstore in memory")
        return

    # Create a folder for this PDF based on its name
    pdf_name = os.path.splitext(os.path.basename(file_path))[0]
    pdf_vectorstore_path = os.path.join(persist_folder, pdf_name)

    if os.path.exists(pdf_vectorstore_path):
        # Load existing vectorstore
        print(f"Loading existing vectorstore for {file_path}")
        embedding_wrapper = LocalEmbedding()
        vectorstore = Chroma(
            persist_directory=pdf_vectorstore_path,
            embedding_function=embedding_wrapper
        )
        current_file_path = file_path
        print("Vectorstore loaded from disk successfully!")
        return

    # If not exists, create vectorstore
    print(f"Creating vectorstore for: {file_path}")
    current_file_path = file_path

    '''loader = PDFPlumberLoader(file_path)
    docs = loader.load()'''
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()

    if not docs or all(not doc.page_content.strip() for doc in docs):
        print("Warning: PDF appears empty or unreadable")
        return
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". "]
    )

    chunks = []
    for i, doc in enumerate(docs):
        page_chunks = text_splitter.split_text(doc.page_content)
        for j, chunk_text in enumerate(page_chunks):
            chunk_doc = Document(page_content=chunk_text, metadata={"page": i+1, "chunk": j+1})
            chunks.append(chunk_doc)

    print(f"Created {len(chunks)} chunks from {len(docs)} pages")

    # Create and persist vectorstore
    embedding_wrapper = LocalEmbedding()
    vectorstore = Chroma.from_documents(
        chunks,
        embedding_wrapper,
        persist_directory=pdf_vectorstore_path
    )
    vectorstore.persist()
    print(f"Vectorstore created and saved to {pdf_vectorstore_path}")


# ---------------- RAG Query with Ollama ----------------
def rag_query(question: str) -> str:
    global vectorstore
    if vectorstore is None:
        return "Error: Vectorstore not initialized. Please run initialize_vectorstore() first."

    print(f"Processing question: {question}")

    try:
        retrieved_docs = vectorstore.similarity_search(question, k=5)
        if not retrieved_docs:
            return "No relevant documents found."

        # Print top 3 chunks
        for i, doc in enumerate(retrieved_docs[:3]):
            print(f"Top Chunk {i+1} ({len(doc.page_content)} chars):\n{doc.page_content[:300]}...\n{'-'*40}\n")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""You are an information extraction assistant. 
Extract the answer to the question using ONLY the information provided below. 
If the answer is a percentage, number, or date, copy it exactly as written. 
If the answer is not present, say 'Not found in the context.'
Information:
{context}

Question: {question}

Answer:"""

        print("Querying Ollama...")
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "top_p": 0.9, "num_predict": 400, "repeat_penalty": 1.1}
        )

        answer = response['message']['content'].strip()
        return answer

    except Exception as e:
        return f"Error during query: {str(e)}"

# ---------------- Main ----------------
if __name__ == "__main__":
    print("\n--- RAG Local Test ---\n")

    # Hardcode PDF path here
    pdf_path = "C:\\Users\\Dell\\Downloads\\SakshiJadhavResume (1).pdf"

    try:
        initialize_vectorstore(pdf_path)
        print("\nVectorstore ready.\n")

        question = input("Enter your question about the PDF: ").strip()
        answer = rag_query(question)
        print(f"\nFinal Answer:\n{answer}\n")

    except Exception as e:
        print(f"Error: {e}")
