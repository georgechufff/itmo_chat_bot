import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from dotenv import load_dotenv
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

if not load_dotenv():
    raise FileNotFoundError("File '.env' does not exist or is empty.")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)

urls = [
    "https://abit.itmo.ru/program/master/ai",
    "https://abit.itmo.ru/program/master/ai_product",
    'https://job.itmo.ru/ru/catalog?category=1',
]

llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0.5, 
                 api_key=os.getenv("OPENAI_API_KEY"),
                 base_url=os.getenv("BASE_URL"))
prompt = hub.pull("rlm/rag-prompt")
runnable = prompt | llm

loader = WebBaseLoader(urls)
documents = loader.load()

pdfs = [
    '10033-abit.pdf',
    '10130-abit.pdf'
]

pdf_documents = []

for pdf in pdfs:
    loader = PyPDFLoader(pdf)
    pdf_documents.append(loader.load())
    
# documents += pdf_documents

print(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

db = FAISS.from_documents(all_splits, embeddings)

st.title("ИТМО")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": 'Привет! Я помогу тебе разобраться с учебными планами магистратур ИТМО. Задавай вопросы!'}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Введите Ваше сообщение."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        query = st.session_state.messages[-1]['content']
        content = db.similarity_search_with_score(query)
        print(content)
        docs_content = "\n\n".join(doc[0].page_content for doc in content)
        print(docs_content)
        answer = runnable.invoke({"question": query, "context": docs_content})
        print(answer)
        response = st.write_stream([s + ' ' for s in answer.content.split()])

    st.session_state.messages.append({"role": "assistant", "content": response})