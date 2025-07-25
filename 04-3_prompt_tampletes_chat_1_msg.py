#######################################
#
# Prompt tampletes - chat - 1 message
#
#######################################

# importar prompt tampletes de chat
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_template('Essa é a minha dúvida: {duvida}')

print(chat_template.format_messages(duvida='Que dia é hoje?'))