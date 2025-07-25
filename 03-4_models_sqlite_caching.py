from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path='caching/langchaing_cache_db.sqlite'))

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

mensagens = [
    SystemMessage(content='Você é um assistente engraçado'),
    HumanMessage(content='Quanto é 1 + 1?')
]

resposta = chat.invoke(mensagens)

print(resposta.content)

resposta = chat.invoke(mensagens)

print(resposta.content)