# Chatbot
I tested out a POC/MVP of using a Chatbot inplace of documentation. One of the biggest problems that ChatGPT has is that it is out of date, or not trained specifically on the documentation that you need. I wanted to see how well it could perform if the documentation was vectorized and persisted and fed to the ChatGPT API and used as a Question and Answering Bot. The results were generally very good and I think this could be a great usecase of LLMs.  

## Create the Vectorstore
1. gather the urls and place them into the data.py file.
2. run _create_vectorstore() on main.py
## Run the Chatbot
1. get your OpenAI api key and either place it in a config file or replace the api_key variable.
2. change your chroma_db_dir variable to a desirable path.
3. run main() and interact with your chatbot!
