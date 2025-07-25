from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#document loading
caminho = 'arquivos/Explorando o Universo das IAs com Hugging Face.pdf'
loader = PyPDFLoader(caminho)
paginas = loader.load()

# text splitting
recur_split  = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', ' ', '']
)

documents = recur_split.split_documents(paginas)

# embedding
embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

# criar a vector store (fazer apenas uma vez)
diretorio = 'arquivos/chroma_vectorstore'

# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embedding_model,
#     persist_directory=diretorio
# )

# print(vectorstore._collection.count())

# carregar (load) a base de dados já salva (para não criar novamente)
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=diretorio
)
print(vectorstore._collection.count())

# retrieval
pergunta = 'O que é o huggingface?'

docs = vectorstore.similarity_search(pergunta, k=5) #k=5 procurar os 5 documentos mais próximos
print(len(docs))