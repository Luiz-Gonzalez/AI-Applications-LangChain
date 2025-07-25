from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

#traduzir de uma lingua para o português
prompt_traducao = ChatPromptTemplate.from_template('Traduza o seguinte texto do {lingua} para o português: {texto}')
chain1 = prompt_traducao | chat | StrOutputParser()
texto_traduzido = chain1.invoke({'lingua' : 'Italiano', 'texto' : 'Ho tanta voglia di andare a vivere in Italia! Ci ho visuto 3 anni fa a Verona e mi è piaciuto tantissimo!'})

# resumir o texto traduzido
prompt_resumo = ChatPromptTemplate.from_template('Resuma o texto em 6 palavras: {texto_para_resumir}')
chain2 = prompt_resumo | chat | StrOutputParser()
texto_resumido = chain2.invoke({'texto_para_resumir' : texto_traduzido})

print(f'Texto traduzido: {texto_traduzido}')
print(f'Texto resumido: {texto_resumido}')

# combinar as duas chains acima
chain3 = chain1 | chain2
combinacao = chain3.invoke({'lingua' : 'Italiano', 'texto' : 'Ho tanta voglia di andare a vivere in Italia! Ci ho visuto 3 anni fa a Verona e mi è piaciuto tantissimo!'})

print(f'Combinacao: {combinacao}')