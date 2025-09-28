from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader


from langchain.docstore.document import Document
from embedding import LocalEmbedding
import os

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
