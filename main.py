# main.py
from vectorstore_manager import initialize_vectorstore
from rag_query import rag_query

if __name__ == "__main__":
    print("\n--- RAG Test ---\n")
    pdf_path = "C:\\Users\\Dell\\Downloads\\SakshiJadhavResume (1).pdf"

    try:
        # Initialize vectorstore first
        initialize_vectorstore(pdf_path)
        print("\nVectorstore ready.\n")

        # Example query
        question = "tell me about her projects and her education?"
        answer = rag_query(question)
        print(f"\nFinal Answer:\n{answer}\n")

    except Exception as e:
        print(f"Error in main block: {e}")
