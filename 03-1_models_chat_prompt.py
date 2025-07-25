from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# prompt few-shot
mensagens = [
    HumanMessage(content='Quanto é 1 + 1?'),
    AIMessage(content='2'),
    HumanMessage(content='Quanto é 10 * 5?'),
    AIMessage(content='50'),
    HumanMessage(content='Quanto é 10 + 3?')
]

resposta = chat.invoke(mensagens)
print(resposta.content)