#######################################
#
# Prompt tampletes - composing
#
#######################################

# importar prompt templates
from langchain.prompts import PromptTemplate
# importar llm
from langchain_openai.llms import OpenAI

llm = OpenAI()

# prompt 1
word_count_template = PromptTemplate.from_template(''''
Responda a pergunta em até {n_words} palavras.
''')
# prompt 2
language_template = PromptTemplate.from_template('''
Retorne a resposta em {language}.
''')
# composing
final_template = (
    word_count_template
    + language_template
    + 'Você deve responder a pergunta (não é para tarduzir a pergunta, é para responder): {question}'

)
final_prompt = final_template.format(n_words=10, language='Inglês', question='O que é uma estrela?')

response = llm.invoke(final_prompt)

print(response)
