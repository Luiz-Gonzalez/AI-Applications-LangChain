from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_template('Crie uma frase sobre o seguinte assunto: {assunto}')

chain = prompt | model | StrOutputParser()

resultado = chain.invoke({'assunto' : 'futebol'})

print(resultado)