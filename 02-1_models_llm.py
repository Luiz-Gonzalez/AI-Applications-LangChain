#######################################
#
# Models - Acessing the language models
#
#######################################
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
import os

# load api key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# loading LLM
llm = OpenAI(api_key=openai_api_key)

# stream calling - stream()
pergunta = 'Conte uma história sobre aprender a programar'
for trecho in llm.stream(pergunta):
    print(trecho, end='')

# várias perguntas em paralelo - batch()
perguntas = [
    'o que é o céu?',
    'o que é a terra?',
    'o que são as estrelas?',
]
respostas = llm.batch(perguntas)
print(respostas)