##############################
#
# RAG - Document Loader (CSV)
#
##############################

from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
from pprint import pprint
# responde perguntas de um documento
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

caminho = 'arquivos/Top 1000 IMDB movies.csv'
loader = CSVLoader(caminho)
documentos = loader.load()

# cada documento vai ser uma linha do CSV
# print(documentos[0].page_content)

# fazer uma pergunta sobre o documento
chain = load_qa_chain(llm=chat, chain_type='stuff')

pergunta = 'Qual é o filme com maior metascore?'
resposta = chain.run(input_documents=documentos[:10], question=pergunta, verbose=False)
print (resposta)