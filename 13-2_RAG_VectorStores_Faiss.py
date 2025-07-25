from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
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

# vector store - sem salvar em disco
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

# retrieval
pergunta = 'O que é o huggingface?'

docs = vectorstore.similarity_search(pergunta, k=5) #k=5 procurar os 5 documentos mais próximos
print(len(docs))

# salvando os dados em disco
vectorstore.save_local('arquivos/faiss_db')

# ler (importar) a base de dados já criada
vectorstore = FAISS.load_local(
    'arquivos/faiss_db',
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)