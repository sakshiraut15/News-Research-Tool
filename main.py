import os
import streamlit as st
import pickle
import time

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


# Load OpenAI API Key from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


st.title("RockyBot: News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")


file_path = "faiss_store_openai.pkl"


main_placeholder = st.empty()


# OpenAI LLM
llm = OpenAI(
    temperature=0.9,
    max_tokens=500,
    api_key=st.secrets["OPENAI_API_KEY"]
)


if process_url_clicked:

    # Remove empty URLs
    urls = [url for url in urls if url.strip()]


    if len(urls) == 0:
        st.warning("Please enter at least one URL")
        st.stop()


    # Load article data
    loader = UnstructuredURLLoader(
        urls=urls
    )


    main_placeholder.text(
        "Data Loading...Started...✅✅✅"
    )


    data = loader.load()



    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ".",
            ","
        ],
        chunk_size=1000,
        chunk_overlap=200
    )


    main_placeholder.text(
        "Text Splitter...Started...✅✅✅"
    )


    docs = text_splitter.split_documents(data)



    # Create embeddings
    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["OPENAI_API_KEY"]
    )


    vectorstore_openai = FAISS.from_documents(
        docs,
        embeddings
    )


    main_placeholder.text(
        "Embedding Vector Started Building...✅✅✅"
    )


    time.sleep(2)



    # Save FAISS index

    with open(file_path, "wb") as f:
        pickle.dump(
            vectorstore_openai,
            f
        )


    st.success(
        "URLs processed successfully ✅"
    )




query = main_placeholder.text_input(
    "Question:"
)



if query:

    if os.path.exists(file_path):

        with open(file_path, "rb") as f:

            vectorstore = pickle.load(f)



        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )



        result = chain.invoke(
            {
                "question": query
            }
        )



        st.header("Answer")


        st.write(
            result["answer"]
        )



        sources = result.get(
            "sources",
            ""
        )



        if sources:

            st.subheader(
                "Sources:"
            )


            sources_list = sources.split("\n")


            for source in sources_list:
                st.write(source)



    else:

        st.warning(
            "Please process URLs first"
        )