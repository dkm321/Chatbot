from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chain_api import ChainAPI

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from config import api_key

embeddings = OpenAIEmbeddingFunction(api_key=api_key)

k = 3
chain_type = 'map_reduce'
return_source_doc = True
persist_dir = '/Users/dk/Projects/chatbot/Chatbot/chroma_test'
user_id = 1
chain = ChainAPI(embeddings, chain_type, k, return_source_doc, persist_dir, user_id)
app = FastAPI()

# Optional: Add CORS middleware if your frontend is on a different domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask/")
async def ask_question(question):
    try:
        response = chain.ask(question)
        return response['answer']
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl/")
async def crawl(base_url):
    try:
        urls = chain.crawl(base_url)
        return urls
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add/")
async def add_to_db(urls):
    try:
        chain.add_to_db([urls])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # To run the app directly from this script.
    # However, for production, use Uvicorn directly.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
