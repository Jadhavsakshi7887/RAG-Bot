# rag_query.py
import ollama
import vectorstore_manager

def rag_query(question: str) -> str:
    """
    Query the RAG system using the initialized vectorstore.
    """
    if vectorstore_manager.vectorstore is None:
        return "Error: Vectorstore not initialized. Please run initialize_vectorstore() first."

    vectorstore = vectorstore_manager.vectorstore
    print(f"Processing question: {question}")

    try:
        # Retrieve top 5 chunks
        retrieved_docs = vectorstore.similarity_search(question, k=5)
        if not retrieved_docs:
            return "No relevant documents found."

        # Print top 3 chunks
        for i, doc in enumerate(retrieved_docs[:3]):
            print(f"Top Chunk {i+1} ({len(doc.page_content)} chars):\n{doc.page_content[:300]}...\n{'-'*40}\n")

        # Combine chunks as context
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
