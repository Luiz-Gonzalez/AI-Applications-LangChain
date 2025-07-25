####################################
#
# Output Parsers - Chat - String
#
####################################

# Passos:
# 1 - Criar o chat template
# 1.1 - Importar o langchain.prompts/ChatPromptTemplate
# 1.2 - Criar um chat_template para ser utilizado
# 1.3 - Invocar o chat_template passando as variáveis
#
# 2  - Importar o modelo/chat a ser utilizado
# 2.1 - Importar o Chat model (ChatOpenAi)do langchain_openai 
# 2.2 - Instanciar o chat -> chat = ChatOpenAi()
#
# 3 - Invocar o chat passando o prompt criado com o template
#
# 4 - Importar o output_parser do langchain core
# 4.1 - instanciar o output_parser (output_parser = StrOutputParser())
# 4.2 - Invokar passando a resposta

from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# cria um prompt de chat baseado em uma lista de mensagens. Cada mensagem é representada por uma tupla
chat_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'Você é um assistente engraçado e se chama {nome_assistente}'),
        ('human', '{pergunta}')
    ]
)

# forma mais aconselhada de chamar: usar invoke ao invés do format_messages
prompt = chat_template.invoke({'nome_assistente' : 'luigi', 'pergunta' : 'Qual o seu nome?'})
resposta = chat.invoke(prompt)

# usar o StrOutputParser para pegar apenas o texto da resposta
# ao inves de ter uma resposta como uma classe AIMessage ter um texto
output_parser = StrOutputParser()
convertido = output_parser.invoke(resposta)

print(convertido)