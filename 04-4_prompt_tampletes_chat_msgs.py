###########################################
#
# Prompt tampletes - chat - messages (list)
#
###########################################

# importar prompt tampletes de chat
from langchain.prompts import ChatPromptTemplate
# importar o modelo de mensagem
from langchain_openai.chat_models import ChatOpenAI
# instanciar o modelo
chat = ChatOpenAI()

# lista com as mensagens, utiliza-se tuplas
chat_template = ChatPromptTemplate.from_messages(
    [
    ('system', 'Você é um assistente engraçado e se chama {nome_assistente}'),
    ('human', 'Olá como vai?'),
    ('ai', 'Melhor agora, como posso ajudá-lo?'),
    ('human', '{pergunta}')
    ]
)

resposta = chat.invoke(chat_template.format_messages(nome_assistente='Luigi', pergunta='Qual o seu nome?'))
print(resposta.content)