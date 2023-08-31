from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import chromadb

import os
import uuid

from crawler import crawl_website
from config import api_key


os.environ['OPENAI_API_KEY'] = api_key

class ChainAPI:

    def __init__(self, embeddings, chain_type, k, return_source_doc, persist_dir, user_id):
        self.embeddings = embeddings
        self.chain_type = chain_type
        self.k = k
        self.return_source_doc = return_source_doc
        self.persist_dir = persist_dir
        self.user_id = user_id
        self.collection_name = f'{user_id}_coll'
        self.chunk_size = 8000
        self.chunk_overlap = 3000
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.db = self.get_vectorstore()

    def create_docs(self, urls):
        # load documents
        loader = UnstructuredURLLoader(urls)
        documents = loader.load()

        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_docs = text_splitter.split_documents(documents)

        return split_docs

    def get_vectorstore(self):
        collection = self.client.get_or_create_collection(name=self.collection_name, embedding_function=self.embeddings)
        return collection
    
    def add_documents(self, collection, documents):
        ids = [str(uuid.uuid1()) for _ in documents]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.page_content for doc in documents]
        collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=texts
        )

    def create_chain(self, chain_type, k, return_source_doc):
        # define retriever
        langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=OpenAIEmbeddings()
        )

        retriever = langchain_chroma.as_retriever(search_type="similarity", search_kwargs={"k": k})

        # Define Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='question', output_key='answer')
        
        # create a chatbot chain. Memory is managed externally.
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0), 
            chain_type=chain_type, 
            retriever=retriever, 
            memory=memory,
            return_source_documents=return_source_doc
        )
        return qa_chain 

    def crawl(self, domain: str):
        cleaned_url = domain.replace("http://", "").replace("https://", "")
            
        start_url = f"http://{cleaned_url}"
        urls = crawl_website(start_url, cleaned_url)

        return urls

    def add_to_db(self, urls):
        split_docs = self.create_docs(urls)
        self.add_documents(self.db, split_docs)

    def ask(self, question):
        chat_history = []
        qa_chain = self.create_chain(self.chain_type, self.k, self.return_source_doc)
        response = qa_chain({"question": question, "chat_history": chat_history})

        return response