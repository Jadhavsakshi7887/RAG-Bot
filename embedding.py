# embedding.py
from sentence_transformers import SentenceTransformer

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
