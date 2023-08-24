from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import os

from data import urls
from config import api_key

chroma_db_dir = '~/Documents/QA/chroma_db'

os.environ['OPENAI_API_KEY'] = api_key

def _create_vectorstore(urls, embeddings_class):
    # load documents
    loader = UnstructuredURLLoader(urls)
    documents = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # create vector database from data
    db = Chroma.from_documents(documents=docs, embedding=embeddings_class(), persist_directory=chroma_db_dir)
    db.persist()

    return db

def load_vectorstore(embeddings_class):
    
    db = Chroma.from_documents(documents=[], persist_directory=chroma_db_dir, embedding=embeddings_class())
    print('Vectorstore exists. Retrieving vectorstore.')

    return db


def load_db(chain_type, k):
    db = load_vectorstore(OpenAIEmbeddings)

    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Define Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        memory=memory
    )
    return qa 

def main():
    qa_chain = load_db('stuff', 3)

    print('\n')

    while True:
        user_input = input('Ask your question: ')
    
        if user_input.lower() in ("quit", 'exit', 'stop'):
            break

        output = qa_chain({"question": user_input})
        print('\n')
        print(output['answer'])
        print('---------------------------------------------------------------')


main()
# _create_vectorstore(urls, OpenAIEmbeddings)