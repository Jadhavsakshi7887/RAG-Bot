
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
import ollama
from langchain.docstore.document import Document

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="D:/huggingface_models"
)

vectorstore = None
current_file_path = None

def initialize_vectorstore(file_path: str):
    global vectorstore, current_file_path

    if vectorstore is not None and current_file_path == file_path:
        print("Using existing vectorstore")
        return

    print(f"Initializing vectorstore for: {file_path}")
    current_file_path = file_path

    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    if not docs or all(not doc.page_content.strip() for doc in docs):
        print("Warning: PDF appears empty or unreadable")
        return
    
    for i, doc in enumerate(docs):
        page_text = doc.page_content.strip()
        print(f"--- Page {i} ---")
        print(f"Length: {len(page_text)} characters")
        print(f"Start of page:\n{page_text[:500]}")  
        print(f"End of page:\n{page_text[-500:]}\n")  
        print("-"*80)

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
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} (Page {chunk.metadata['page']}, Chunk {chunk.metadata['chunk']}) ---")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content:\n{chunk.page_content}")
        print("-"*80)

    class LocalEmbedding:
        def embed_documents(self, texts):
            return embed_model.encode(texts, show_progress_bar=False, convert_to_tensor=False).tolist()
        def embed_query(self, text):
            return embed_model.encode([text], convert_to_tensor=False).tolist()[0]

    embedding_wrapper = LocalEmbedding()

    try:
        vectorstore = Chroma.from_documents(chunks, embedding_wrapper)
        print("Vectorstore initialized successfully!")
    except Exception as e:
        print(f"Error creating vectorstore: {e}")

def rag_query(question: str) -> str:
    if vectorstore is None:
        return "Error: Vectorstore not initialized. Please run initialize_vectorstore() first."

    print(f"Processing question: {question}")

    try:
        retrieved_docs = vectorstore.similarity_search(question, k=5)

        if not retrieved_docs:
            return "No relevant documents found."

        for i, doc in enumerate(retrieved_docs[:3]):
            print(f"Top Chunk {i+1} ({len(doc.page_content)} chars):\n{doc.page_content[:300]}...\n{'-'*40}\n")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Prepare prompt
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
            options={
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 400,
                "repeat_penalty": 1.1
            }
        )

        answer = response['message']['content'].strip()
        print(f"Generated answer ({len(answer)} chars): {answer[:200]}...")
        return answer

    except Exception as e:
        error_msg = f"Error during query: {str(e)}"
        print(error_msg)
        return error_msg
if __name__ == "__main__":
    print("\n--- Gemini RAG Test ---\n")
    pdf_path = "C:\\Users\\Dell\\Downloads\\SakshiJadhavResume (1).pdf"
    try:
        initialize_vectorstore(pdf_path)
        print("\nVectorstore ready.\n")
        question = "What is the SSC; Percentage of Sakshi Jadhav?"
        answer = rag_query(question)
        print(f"\nFinal Answer:\n{answer}\n")
    except Exception as e:
        print(f"Error in main block: {e}")

