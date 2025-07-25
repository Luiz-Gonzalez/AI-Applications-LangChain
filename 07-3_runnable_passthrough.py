from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# nome do produto
prompt_nome = ChatPromptTemplate.from_template('Crie um nome para o seguinte produto: {produto}. Não escreva nada a mais a não ser o nome.')
chain_nome = prompt_nome | chat | StrOutputParser()
# cliente potencial
prompt_cliente = ChatPromptTemplate.from_template('Descreva, de forma resumida, o ciente potencial para o seguinte produto: {produto}')
chain_cliente = prompt_cliente | chat | StrOutputParser()

prompt = ChatPromptTemplate.from_template('''Dado o produto com o seguinte nome e seguinte
público pontencial, desenvolva um anúncio para o produto.

Nome do produto: {nome_produto}
Característica do produto: {produto}
Público: {publico}
''')

parallel = RunnablePassthrough().assign(**{'nome_produto' : chain_nome, 'publico' : chain_cliente})

chain = parallel | prompt | chat | StrOutputParser()

resposta = chain.invoke({'produto' : 'Curso para ensinar o cachorro a não comer mais cocô'})

pprint(resposta)
