import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 2) Prepare documents
docs = [
    "Kaggle competitions evaluate data science skills.",
    "RAG improves LLM answers using external knowledge.",
    "FAISS is a library for efficient similarity search.",
    "LLaMA can run locally using llama-cpp."
]

# 3) Create embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, convert_to_numpy=True)

# 4) Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

def rag_query(query, k=2):
    # Embed query
    q_emb = embedder.encode([query], convert_to_numpy=True)

    # Retrieve top-k docs
    D, I = index.search(q_emb, k)
    context = "\n".join([docs[i] for i in I[0]])

    # Compose prompt
    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question: {query}
Answer:
"""
    # Generate
    out = llm(prompt,200)
    return out

# Example
print(rag_query("What is RAG and why is it useful?"))
