from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from pprint import pprint
from langchain_openai import ChatOpenAI

load_dotenv()

# carregar e juntar documentos
caminhos = [
    'arquivos/Explorando a API da OpenAI.pdf',
]
## document loading
paginas = []
for caminho in caminhos:
    loader = PyPDFLoader(caminho)
    paginas.extend(loader.load())
## text splitting
recur_plit = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=['\n\n', '\n', '.', ' ', '']
)
documents = recur_plit.split_documents(paginas)
## alterar os metadados
for i, doc in enumerate(documents):
    doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '')
    doc.metadata['doc_id'] = i

# vector store
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings()
)

# prompt
prompt = ChatPromptTemplate.from_template(
    '''Responda as perguntas se baseando no contexto fornecido.
    
    Contexto: {contexto}
    Pergunta: {pergunta}'''
)

# retriever
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k' : 5, 'fetch_k' : 25}) #transforma em um runnable


# Funcao para arrumar o que volta para tirar o que não é necessário (metada, etc) e deixar só o texto
# Dessa forma fica muito mais facil para a LLM entender.
def join_documents(input):
    # para cada objeto 'c' na lista de objetos 'contexto' vou pegar o page_content 'c.page_content' e adicionar '\n\n'
    # cada 'c' já é um objeto document da lista. O retorno da funcao vai ser todos os textos juntos uns aos poutros.
    # 
    # Como é a estrutura:  
    #     input['contexto'] = [
    #     Document(page_content="Texto 1", metadata={...}),
    #     Document(page_content="Texto 2", metadata={...}),
    #     Document(page_content="Texto 3", metadata={...})
    #     ]
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

# setup Runnable com funcao para limpar o texto
setup = RunnableParallel({
    'pergunta' : RunnablePassthrough(),
    'contexto' : retriever
    
}) | join_documents #quando já temos um runnable podemos adiconar o pipe |

# chain

chain = setup | prompt | ChatOpenAI()
resposta = chain.invoke('O que é a OpenAI?')
print(resposta.content)