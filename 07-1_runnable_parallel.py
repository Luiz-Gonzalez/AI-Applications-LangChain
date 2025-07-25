from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# nome do produto
prompt_nome = ChatPromptTemplate.from_template('Crie um nome para o seguinte produto: {produto}')
chain_nome = prompt_nome | chat | StrOutputParser()
# cliente potencial
prompt_cliente = ChatPromptTemplate.from_template('Descreva o ciente potencial para o seguinte produto: {produto}')
chain_cliente = prompt_cliente | chat | StrOutputParser()

# executar duas chains em paralelo
parallel = RunnableParallel({'nome_produto' : chain_nome, 'publico' : chain_cliente})

# anúncio: juntar as duas chains anteriores em uma para criar o anúncio
prompt_anuncio = ChatPromptTemplate.from_template('''Dado o produto com o seguinte nome e seguinte
público potencial, desenvolva um anúncio curto para Instagram para o produto.

Nome do produto: {nome_produto}
Público: {publico}''')
chain_anuncio = parallel | prompt_anuncio | chat | StrOutputParser()
resposta = chain_anuncio.invoke({'produto' : 'Um Copo inquebrável'})
print(resposta)