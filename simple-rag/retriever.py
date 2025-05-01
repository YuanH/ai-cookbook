from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, docs):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = docs
        self.embeddings = self.model.encode(docs, convert_to_numpy=True)
        dim = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query, k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        return [self.docs[i] for i in indices[0]]