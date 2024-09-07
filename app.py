import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

# create/ allocate  memory to remeber conversation context
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain

from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI


# from htmlTemplates import css, bot_template, user_template

import os 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs: # loop through pdf docs
        pdf_reader = PdfReader(pdf) # reads each page from pdf doc
        for page in pdf_reader.pages:  # loop through each page
            text += page.extract_text() # extract text from each page

    return text


def get_text_chunks(text):
    # we will use langchain to split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator = "\n", 
        chunk_size = 1000, 
        chunk_overlap = 200,
        length_function = len 
    )
    
    chunks = text_splitter.split_text(text)
    return chunks
    

def get_vector_store(text_chunks):

    # os.environ["OPENAI_API_KEY"] = ""
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    # vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    vector_store = Chroma.from_texts(text_chunks, embedding = embeddings)
    return vector_store





def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}, 
    #                      huggingfacehub_api_token = "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")
    
    # llm.client.api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(""" # QUESTION :zany_face: """)
            st.write(message.content)
        else:
            st.write(""" # BOT RESPOSNE :robot_face: """)
            st.write(message.content)


def main():

    load_dotenv()

    st.set_page_config(page_title="Chat with your PDF documents", page_icon=":books:")

    # st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with mulitple pdfs :books:")
    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_user_input(user_question)


    # st.write(user_template.replace("{{MSG}}", "hello bot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # if button is clicked do the following
                # get the raw pdf text
                # get text chunks
                # create vector store using faiss-cpu


                # get raw text from pdf
                raw_text = get_pdf_text(pdf_docs)
                
                # get text chunks to return a list of chunks from pdf text
                text_chunks = get_text_chunks(raw_text)
               
                # create vector store in faiss-cpu using openai embeddings
                vector_store = get_vector_store(text_chunks)

                # create a conversation chain

                # make conversation persistent, prevent reinitialization of variable
                st.session_state.conversation = get_conversation_chain(vector_store)




if __name__ == "__main__":
    main()
