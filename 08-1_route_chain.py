from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pprint import pprint
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

# prompts/chains específicos de cada área
prompt_matematica = ChatPromptTemplate.from_template('''Você é um professor de Matemática
do ensino fundamental capaz de dar respostas muito detalhadas e didáticas. Responda a seguinte pergutna de um aluno:
Pergunta: {pergunta}''')
chain_matematica = prompt_matematica | model

prompt_fisica = ChatPromptTemplate.from_template('''Você é um professor de Física
do ensino fundamental capaz de dar respostas muito detalhadas e didáticas. Responda a seguinte pergutna de um aluno:
Pergunta: {pergunta}''')
chain_fisica = prompt_fisica | model

prompt_historia = ChatPromptTemplate.from_template('''Você é um professor de História
do ensino fundamental capaz de dar respostas muito detalhadas e didáticas. Responda a seguinte pergutna de um aluno:
Pergunta: {pergunta}''')
chain_historia = prompt_historia | model

prompt_generico = ChatPromptTemplate.from_template('''{pergunta}''')
chain_generica = prompt_generico | model

# classificador da pergunta
prompt_categorizador = ChatPromptTemplate.from_template('Você deve categorizar a seguinte pergunta: {pergunta}')

class Categorizador(BaseModel):
    '''Categoriza as perguntas de alunos do ensino fundamental'''
    area_conhecimento: str = Field(description='Área de conhecimento da pergunta feita pelo aluno. \
    Deve ser "física", "matemática" ou "história". Caso não se encaixe em nenhuma delas, retorne "outra".')

model_estruturado = prompt_categorizador | model.with_structured_output(Categorizador)

# roteador - função
def route(input):
    if input['categoria'].area_conhecimento == 'matemática':
        return chain_matematica
    if input['categoria'].area_conhecimento == 'física':
        return chain_fisica
    if input['categoria'].area_conhecimento == 'história':
        return chain_historia
    return chain_generica

#estrutura para o roteador
chain = RunnablePassthrough().assign(categoria=model_estruturado) | route
resposta = chain.invoke({'pergunta' : 'Por que ocorreu a unificação da Itália?'})
pprint(resposta.content)