from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint

chunk_size = 300
chunk_overlap = 30

char_split = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

caminho = 'arquivos/Explorando o Universo das IAs com Hugging Face.pdf'

loader = PyPDFLoader(caminho)
docs = loader.load()

splits = char_split.split_documents(docs)

pprint(splits)