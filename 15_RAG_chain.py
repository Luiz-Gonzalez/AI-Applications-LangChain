from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pprint import pprint
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# importar os dados
caminhos = [
    'arquivos/Explorando o Universo das IAs com Hugging Face.pdf',
    'arquivos/Explorando a API da OpenAI.pdf'
]

paginas = []

for caminho in caminhos:
    loader = PyPDFLoader(caminho)
    paginas.extend(loader.load())

# fazer o splitt dos dados

recur_split = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=['\n\n', '\n', '.', ' ', '']
)

documents = recur_split.split_documents(paginas)

# formatar metadata dos documentos carregados
for i, doc in enumerate(documents):
    doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '')
    doc.metadata['doc_id'] = i

# criar a vector store
diretorio = 'arquivos/chat_retrieval_db'

embedding_model = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=diretorio,
)

#criar estrutura de conversa
chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# esta forma já tem um prompt padrão da chain para ligar a pergunta aos documentos
chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectordb.as_retriever(search_type='mmr'),
)

pergunta = 'O que é huggingface e como faço para acessá-lo?'

resposta = chat_chain.invoke({'query' : pergunta})

print(resposta)
print('\n\n=======================\n\n')

# Exemplo alterando o prompt da chain
chain_prompt = PromptTemplate.from_template(
'''Utilize o contexto fornecido para responder a pergunta ao final.
Se você não sabe a resposta, apenas diga que não sabe e não tente inventar a resposta.
Utilize três frases no máximo, mantenha a resposta consisa.

Contexto: {context}

Pergunta: {question}

Resposta: 
''')

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectordb.as_retriever(search_type='mmr'),
    chain_type_kwargs={'prompt' : chain_prompt},
    return_source_documents=True,
)

pergunta = 'O que é huggingface e como faço para acessá-lo?'

resposta = chat_chain.invoke({'query' : pergunta})

print(resposta)


##### outros chains types

# STUFF
# chat_chain = RetrievalQA.from_chain_type(
#     llm=chat,
#     retriever=vectordb.as_retriever(search_type='mmr'),
#     chain_type='stuff',
# )

# MAP REDUCE
# chat_chain = RetrievalQA.from_chain_type(
#     llm=chat,
#     retriever=vectordb.as_retriever(search_type='mmr'),
#     chain_type='map_reduce',
# )

# REFINE
# chat_chain = RetrievalQA.from_chain_type(
#     llm=chat,
#     retriever=vectordb.as_retriever(search_type='mmr'),
#     chain_type='refine',
# )