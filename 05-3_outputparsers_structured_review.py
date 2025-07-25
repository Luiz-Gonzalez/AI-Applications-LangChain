from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

review_cliente = """Este soprador de folhas é bastante incrível. Ele tem 
quatro configurações: sopro de vela, brisa suave, cidade ventosa 
e tornado. Chegou em dois dias, bem a tempo para o presente de 
aniversário da minha esposa. Acho que minha esposa gostou tanto 
que ficou sem palavras. Até agora, fui o único a usá-lo, e tenho 
usado em todas as manhãs alternadas para limpar as folhas do 
nosso gramado. É um pouco mais caro do que os outros sopradores 
de folhas disponíveis no mercado, mas acho que vale a pena pelas 
características extras."""

class Review(BaseModel):
    '''Avalia review do cliente'''
    presente: bool = Field(description='True se foi para presente e Fasle se não foi')
    dias_entrega: int = Field(description='Quantos dias para a entrega do produto')
    percepcao_valor: list[str] = Field(description='Extraia qualquer frase sobre o valor ou preço do produto. Retorne uma lista.')

llm_estruturada = chat.with_structured_output(Review)
resposta = llm_estruturada.invoke(review_cliente)
pprint(resposta)