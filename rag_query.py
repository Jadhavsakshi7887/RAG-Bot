
from langchain_openai import ChatOpenAI
import vectorstore_manager
import os

llm = ChatOpenAI(
    model="openai/gpt-5",  
    openai_api_key=os.getenv("OPENROUTER_API_KEY", "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"),
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=400,
    temperature=0.3,
    top_p=0.9
)

def rag_query(question: str) -> str:
    if vectorstore_manager.vectorstore is None:
        return "Error: Vectorstore not initialized."

    vectorstore = vectorstore_manager.vectorstore
    retrieved_docs = vectorstore.similarity_search(question, k=10)
    if not retrieved_docs:
        return "No relevant documents found."

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are an information extraction assistant. Use ONLY the provided context."
             " If the answer is not present, say 'Not found in the context.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ])
        return response.content.strip()
    except Exception as e:
        return f"Error during query: {str(e)}"
