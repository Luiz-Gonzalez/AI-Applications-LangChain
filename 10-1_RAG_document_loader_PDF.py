##############################
#
# RAG - Document Loader (PDF)
#
##############################

from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders.pdf import PyPDFLoader
from dotenv import load_dotenv
from pprint import pprint
# responde perguntas de um documento
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# caminho do arquivo
caminho = 'arquivos/Explorando o Universo das IAs com Hugging Face.pdf'
# estanciar o loader de pdf como o caminho do documento
loader = PyPDFLoader(caminho)
# carregar o documento
# cada página vai ser um documento. Acessamos por documento[0]: primeira página, etc
documentos = loader.load()

#quantidade de páginas
# print(len(documentos))

# imprimir uma página
# print(documentos[2].page_content)

# metadados
# print(documentos[2].metadata)

chain = load_qa_chain(llm=chat, chain_type='stuff', verbose=True) #verbose mostra tudo que está sendo feito por debaixo dos panos

pergunta = 'Quais assuntos são tratados no documento?'

reposta = chain.run(input_documents=documentos[:6], question=documentos)

pprint(reposta)


