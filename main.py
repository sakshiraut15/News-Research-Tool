import os
import time
import pickle
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


# Load API Key from Streamlit Secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


st.set_page_config(
    page_title="RockyBot: News Research Tool",
    page_icon="📈"
)

st.title("RockyBot: News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)


if process_url_clicked:

    urls = [url for url in urls if url.strip()]

    if len(urls) == 0:
        st.warning("Please enter at least one URL.")
        st.stop()

    main_placeholder.text("Loading articles...")

    loader = UnstructuredURLLoader(urls=urls)

    data = loader.load()

    main_placeholder.text("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(data)

    main_placeholder.text("Creating embeddings...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings,
    )

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    st.success("URLs processed successfully ✅")


query = st.text_input("Question:")

if query:

    if not os.path.exists(file_path):
        st.warning("Please process URLs first.")
        st.stop()

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )

    result = chain.invoke(
        {
            "question": query
        }
    )

    st.header("Answer")

    st.write(result["answer"])

    sources = result.get("sources", "")

    if sources:

        st.subheader("Sources")

        for source in sources.split("\n"):

            if source.strip():

                st.write(source)