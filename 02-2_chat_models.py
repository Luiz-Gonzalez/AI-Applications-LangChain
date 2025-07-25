from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pprint import pprint
import os

# load api key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

chat = ChatOpenAI(api_key=openai_api_key, model='gpt-4.1-nano-2025-04-14')

mensagens = [
    SystemMessage(content='Você é um assistente que conta piadas'), #define o comportamento da LLM
    HumanMessage('Quanto é 1 + 1?')
]

resposta = chat.invoke(mensagens)
pprint(resposta.content)
pprint(resposta.response_metadata['token_usage']['total_tokens'])