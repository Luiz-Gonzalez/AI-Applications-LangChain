from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

class Piada(BaseModel):
    '''Piada para contar ao usuário'''
    introducao: str = Field(description='A introdução da piada')
    punchline: str = Field(description='A conclusão da piada')
    avaliacao: Optional[int] = Field(description='O quão engraçada é a piada de 1 a 10')

# passar a classe Piada() ao método para garatir que sempre retorne
# a saída da mesma forma
llm_estruturada = chat.with_structured_output(Piada)
resposta = llm_estruturada.invoke('Conte uma piada sobre gatinhos')
# retorna uma classe Pydantic de Piada()
pprint(resposta)
pprint(resposta.introducao)
pprint(resposta.punchline)
pprint(resposta.avaliacao)