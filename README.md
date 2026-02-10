# small_rag_project

# What is Rag?
RAG (Retrieval-Augmented Generation) is about supplying external knowledge to the model at query time so it can answer using your data instead of relying only on what it learned during training.

** A RAG system: **
- Searches your documents (PDFs, notes, DB, logs, etc.)
- Retrieves the most relevant pieces
- Injects them into the prompt as context
- Then asks the model to answer using that context
  
Below is a minimal RAG (Retrieval-Augmented Generation) pipeline using a LLaMA model with local inference and a vector store. This example uses:
- llama-cpp-python for LLaMA inference
- sentence-transformers for embeddings
- FAISS for retrieval

It ingests a few text files, builds embeddings, retrieves relevant chunks, and feeds them to LLaMA as context.
pip install llama-cpp-python sentence-transformers faiss-cpu
