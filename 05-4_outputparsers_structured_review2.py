# 1 - Estruturar a classe
# 2 - Instanciar o ChatOpenAi
# 3 - Estruturar o chat
# 4 - Invocar o chat estruturado para resonder


from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

review_cliente = """Este soprador de folhas é ok. Ele tem 
quatro configurações: sopro de vela, brisa suave, cidade ventosa 
e tornado. Chegou atrasado, não a tempo para o presente de 
aniversário da minha esposa. Acho que minha esposa não gostou tanto. Até agora, fui o único a usá-lo, e tenho 
usado em todas as manhãs alternadas para limpar as folhas do 
nosso gramado. É um pouco mais caro do que os outros sopradores 
de folhas disponíveis no mercado, mas acho que vale a pena pelas 
características extras."""

review_cliente2 = """Este soprador de folhas é bastante incrível. Ele tem 
quatro configurações: sopro de vela, brisa suave, cidade ventosa 
e tornado. Chegou em dois dias, bem a tempo para o presente de 
aniversário da minha esposa. Acho que minha esposa gostou tanto 
que ficou sem palavras. Até agora, fui o único a usá-lo, e tenho 
usado em todas as manhãs alternadas para limpar as folhas do 
nosso gramado. É um pouco mais caro do que os outros sopradores 
de folhas disponíveis no mercado, mas acho que vale a pena pelas 
características extras."""

class Avaliacao(BaseModel):
    produto_descricao: str = Field(description='Escreva uma breve descrição do produto')
    entrega: int = Field(description='Avalie criteriosamente quanto o cliente ficou satisfeito com a entrega. Forneça uma nota de 0 a 5')
    produto: int = Field(description='Avalie criteriosamente quanto o cliente ficou satisfeito com o produto. Forneça uma nota de 0 a 5')
    atendimento: int = Field(description='Avalie criteriosamente quanto o cliente ficou satisfeito com o atendimento. Forneça uma nota de 0 a 5.')
    satisfacao: int  = Field(description='Avalie criteriosamente quanto o cliente ficou satisfeito de forma geral com a compra. Forneça uma nota de 0 a 5.')

chat_estruturado = chat.with_structured_output(Avaliacao)

resposta = chat_estruturado.invoke(review_cliente)

print(resposta)