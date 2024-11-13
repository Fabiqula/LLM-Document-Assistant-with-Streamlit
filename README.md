# LLM-Document-Assistant-with-Streamlit
A Streamlit-based app for document summarization and Q&amp;A using OpenAI and Pinecone embeddings.
Supports PDF, DOCX, and TXT files

Document Summarization and Q&A App
This app is a Streamlit-based tool that enables summarization and question-answering on uploaded documents,
including PDF, DOCX, and TXT files.
Leveraging OpenAI's GPT-3.5-turbo model for language processing and Pinecone for vector storage,
the app provides seamless document handling, enabling users to:

Summarize Documents:
Automatically generate concise summaries of large documents using a map-reduce approach for effective insights.
Q&A Functionality:
Ask questions about document content and receive accurate responses, with context from the entire document or selected parts.
Cost Estimation: Calculate token usage and estimated costs for embeddings, ensuring transparency in usage.
Memory and Contextual History: Retain chat history for contextualized question-answering with memory management options.
Features
Multi-Format Support: Uploads PDF, DOCX, and TXT documents.
Interactive Summarization: Configurable chunk size for summarization based on document type.
Conversational Q&A: Built-in memory to handle multi-turn conversations for better contextual answers.
Cost Analysis: Tracks token usage and estimated cost for embeddings and API interactions.
Efficient Embeddings with Pinecone:
Utilizes free Pinecone account for embedding storage, enabling fast document retrieval and question-answering.
It works within Pinecone free account restriction of 1 index, deleting previous upon creating a new one.

Tech Stack
Frontend: Streamlit for user interface
NLP: OpenAI's GPT-3.5-turbo for Q&A and summarization
Vector Storage: Pinecone for scalable, fast embedding storage and retrieval
LangChain: For chaining and processing document chunks
This app is an ideal tool for users looking to gain insights from large documents and interact with content through conversational AI.