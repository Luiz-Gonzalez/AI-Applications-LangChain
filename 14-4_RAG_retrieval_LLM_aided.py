
from langchain_openai.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


# informar explicitamente qual metadados a LLM pode utilizar como filtro
metadata_info = [
    AttributeInfo(
        name='source',
        description='Nome da apostila de onde o texto original foi retirado. Deve ter o valor de: Explorando o Universo das IAs com Hugging Face.pdf ou Explorando a API da OpenAI.pdf',
        type='string'
    ),
    AttributeInfo(
        name='page',
        description='A página da apostila da onde o texto se origina',
        type='integer'
    ),
]
document_description = 'Apostila de cursos'

#LLM
llm = OpenAI()

# Retriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_description,
    metadata_info,
    verbose=False,
)

pergunta = 'Quais detalhes são descritos na página 44 da apostila Explorando a API da OpenAI?'

docs = retriever.get_relevant_documents(pergunta)

for doc in docs:
    print(doc.page_content)
    print(f'==========={doc.metadata}\n\n')