# USER MANUAL RAG for BMW X1

The **BMW X1 Assistant API** is a FastAPI-based application designed to provide intelligent assistance for BMW X1 owners. It leverages advanced natural language processing (NLP) techniques to answer user queries related to the BMW X1 owner's manual. The application uses a vector database for document retrieval and a language model for generating responses.

---

## Features

- **Natural Language Query Handling**: Users can ask questions in plain English.
- **Vector Database Integration**: Efficient document retrieval using embeddings.
- **Customizable Language Model**: Supports local language models for privacy and performance.
- **Multi-Query Retrieval**: Generates multiple perspectives of a query to improve search accuracy.
- **Streamlit Frontend**: A user-friendly interface for interacting with the assistant.
- **CORS Support**: Allows cross-origin requests for flexible integration.

---

## Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [Chroma](https://www.trychroma.com/)
- **Language Model**: [Llama 3.2](https://www.llama.com)[LangChain Ollama](https://www.langchain.com/)
- **Embeddings**: `nomic-embed-text`
