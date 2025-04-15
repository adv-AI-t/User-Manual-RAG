import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever



# Streamlit UI
st.title("BMW X1 Assistant")
st.write("Ask any question related to the BMW X1 owner's manual and get detailed answers.")

# Input field for user query
user_query = st.text_input("Enter your question:", placeholder="How to activate and deactivate the key card?")

if st.button("Submit"):
    if user_query:
        try:

            # Initialize VectorDB and LLM
            vector_db = Chroma(
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="bmw-rag",
    persist_directory="../chroma_db_store",
)

            local_model = "llama3.2"
            llm = ChatOllama(model=local_model)

            QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)


            retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

            template = """
Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

            prompt = ChatPromptTemplate.from_template(template)

            chain = (
    {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)
            #logging the alternate queries
            # alt_queries = QUERY_PROMPT.format(question=user_query)
            # st.write("Generated Queries:", alt_queries)

            #logging the context
            # context = retriever.get_relevant_documents(user_query)
            # st.write("Retrieved Context:", context)

            # Invoke the chain
            response = chain.invoke(user_query)
            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question before submitting.")

st.sidebar.header("Settings")
st.sidebar.write("Make adjustments to the application behavior here.")
