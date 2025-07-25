from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

memory = InMemoryChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Você é um tutor de programação chamado Luigi. Responda as perguntas de forma didática'),
    ('placeholder', '{memoria}'), # só será incluido se {memoria} tiver algo
    ('human', '{pergunta}'),
])

chain = prompt | ChatOpenAI()

# lugar onde será armazenado a memória
store = {}
# função para adicionar novos elemenos ao store
def get_by_session_id(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_com_memoria = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key='pergunta',
    history_messages_key='memoria'
)

config = {'configurable' : {'session_id' : 'usuario_a'}}
resposta = chain_com_memoria.invoke({'pergunta' : 'Meu nome é Luiz'}, config=config)
print(resposta.content)

resposta = chain_com_memoria.invoke({'pergunta' : 'Qual é o meu nome?'}, config=config)
print(resposta.content)