from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

prompt = ChatPromptTemplate.from_template('Fale uma curiosidade sobre o assunto: {assunto}')
chain_curiosidade = prompt | chat | StrOutputParser()

prompt = ChatPromptTemplate.from_template('Crie uma história sobre o seguinte fato curioso: {assunto}')
chain_hitoria = prompt | chat | StrOutputParser()

chain = chain_curiosidade | chain_hitoria

resultado = chain.invoke({'assunto' : 'gatinhos'})

print(resultado)