# small_rag_project
Require Libraries are: !pip install huggingface_hub torch sentence-transformers faiss-cpu
# What is LLM?
Large Language Models (LLMs) are AI systems designed to understand, process and generate human-like text. They are built using advanced neural network architectures that allow them to learn patterns, context and semantics from vast amounts of text data. 

# What is Rnn?
RNN processes sequences sequentially using hidden memory, while transformers process the entire sequence simultaneously using self-attention to capture relationships between all tokens.

# What is word Embedding?
Embeddings assign each word a learned coordinate in a high-dimensional space; the model processes these numbers mathematically and generates text by selecting the vocabulary word whose vector best matches the output vector.

# What is Positional Embedding?
Positional embedding tells the model the order (position) of words in a sentence.

# What is Self Attention?
Self-attention allows each token to weigh and gather information from all other tokens in a sequence, enabling context-aware understanding of language.


Multi-Head Attention means running self-attention multiple times in parallel, each time focusing on different relationships in the sentence.


# What is Rag?
RAG (Retrieval-Augmented Generation) is about supplying external knowledge to the model at query time so it can answer using your data instead of relying only on what it learned during training.

A RAG system:
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
