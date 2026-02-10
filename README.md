# small_rag_project

Below is a minimal RAG (Retrieval-Augmented Generation) pipeline using a LLaMA model with local inference and a vector store. This example uses:
- llama-cpp-python for LLaMA inference
- sentence-transformers for embeddings
- FAISS for retrieval

It ingests a few text files, builds embeddings, retrieves relevant chunks, and feeds them to LLaMA as context.
pip install llama-cpp-python sentence-transformers faiss-cpu
