import langchain
import os
import cohere
import pinecone
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone, FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplate import bot_template, user_template, css

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
llm_model = "gpt-3.5-turbo"
langchain.verbose = False

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def vector_embeddings_store(text_chunks, embeddingtype, vectordb):
    print(type(text_chunks))
    if embeddingtype == "Cohere":
        embeddings = CohereEmbeddings(model="embed-english-light-v2.0", cohere_api_key=os.environ["COHERE_API_KEY"])
        dimension = 1024
        if vectordb == "Pinecone":
            pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
            # doc_result = embeddings.embed_documents(text_chunks)
            # print(type(doc_result))
            index_name = "streamlitqa"
            if index_name in pinecone.list_indexes():
                pinecone.delete_index(index_name)
            pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")
            index = pinecone.Index(index_name)
            vectorstore = Pinecone.from_texts(text_chunks, embeddings, index_name=index_name)
        elif vectordb == "faiss":
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    elif embeddingtype == "OpenAI":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        if vectordb == "Pinecone":
            pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
            # doc_result = embeddings.embed_documents(text_chunks)
            # print(type(doc_result))
            index_name = "streamlitqa"
            if index_name in pinecone.list_indexes():
                pinecone.delete_index(index_name)
            pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")
            index = pinecone.Index(index_name)
            vectorstore = Pinecone.from_texts(text_chunks, embeddings, index_name=index_name)
        elif vectordb == "faiss":
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    # elif embeddingtype == "HuggingFace":
    #   embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def handle_question(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    #st.sidebar.image("bot1.jpg",width=50)
    st.write(css, unsafe_allow_html=True)
    st.title("Langchain Application for PDF Documents")
    question = st.text_input("Please Enter your Question")
    if question:
        handle_question(question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.sidebar.subheader("Your Documents")
    pdf_docs = st.sidebar.file_uploader("Upload PDF files and CLick Proceed", accept_multiple_files=True)
    embedding_type = st.sidebar.selectbox(label="Select the Type of Embedding",
                                          options=["", "OpenAI", "Cohere"])
    print(embedding_type)
    vector_database = st.sidebar.selectbox(label="Select the Vector Database", options=["", "Pinecone", "faiss"])
    print(vector_database)
    if st.sidebar.button("Proceed"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs=pdf_docs)
            # st.write(raw_text)
            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            # st.write(text_chunks)
            # get the embeddings
            vectorstore = vector_embeddings_store(text_chunks, embedding_type, vector_database)
            st.write(vectorstore)
            st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
