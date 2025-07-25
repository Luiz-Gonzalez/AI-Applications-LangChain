from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

caminhos = [
    'arquivos/Explorando o Universo das IAs com Hugging Face.pdf',
    'arquivos/Explorando a API da OpenAI.pdf',
    'arquivos/Explorando a API da OpenAI.pdf',
]
# carregar os documentos
paginas = []

for caminho in caminhos:
    loader = PyPDFLoader(caminho)
    paginas.extend(loader.load())
# fazer o split
recur_split = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', ' ', '']
)
documents = recur_split.split_documents(paginas)

# modificando o metadata (tirar info desnecessária e incluir ID)
for i, doc in enumerate(documents):
    doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '') # limpar o metadata de cada documento para tirar parte inútil
    doc.metadata['doc_id'] = i # inserir um novo dado no metadada, neste caso um doc_id
# criar a vector store
embedding_model = OpenAIEmbeddings()

diretorio = 'arquivos/chroma_retrieval_db'

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=diretorio,
)

# Retrieval - Semantic Search
pergunta = 'O que a apostila de hugging face fala sobre a openai e o chatgpt?'

docs = vectordb.similarity_search(
    pergunta,
    k=3,
    # source única
    # filter={'source' : 'Explorando o Universo das IAs com Hugging Face.pdf'}
    
    # multiplas sources
    # filter={'source' : {'$in':['Explorando o Universo das IAs com Hugging Face.pdf', 'Explorando a API da OpenAI.pdf']}}
    
    # filtrar ultilizando mais de um metadado com vários parâmetros
    filter={'$and':
        [
            {'source' : {'$in':['Explorando o Universo das IAs com Hugging Face.pdf', 'Explorando a API da OpenAI.pdf']}},
            {'page' : {'$in':[12, 13, 14, 15, 16, 17, 18, 19]}}

        ]
    }
)

for doc in docs:
    print(doc.page_content)
    print(f'==========={doc.metadata}\n\n')