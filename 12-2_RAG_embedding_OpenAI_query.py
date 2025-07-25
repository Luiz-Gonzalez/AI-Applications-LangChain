from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

pergunta = 'O que é um cachorro?'
emb_query = embedding_model.embed_query(pergunta)

