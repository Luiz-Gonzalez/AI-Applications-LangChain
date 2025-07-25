#######################################
#
# Prompt tampletes - LLM basics
#
#######################################

# importar uma LLM
from langchain_openai.llms import OpenAI
# importar prompt templates
from langchain.prompts import PromptTemplate

llm = OpenAI()

# criar a casca que pode receber diversos inputs diferentes
prompt_template = PromptTemplate.from_template(''''
Responda a seguinte pergunta do usuário em até {n_palavras} palavras:
{pergunta}       
''', partial_variables={'n_palavras' : 10})

print(prompt_template.format(pergunta='O que é um buraco negro?'))