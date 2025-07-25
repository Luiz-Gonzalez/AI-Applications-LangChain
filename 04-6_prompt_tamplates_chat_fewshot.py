####################################
#
# Prompt tampletes - Chat - few-shot
#
####################################

from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate

chat = ChatOpenAI()

exemplos = [
    {"pergunta": "Quem viveu mais tempo, Muhammad Ali ou Alan Turing?", 
     "resposta": 
     """São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Quantos anos Muhammad Ali tinha quando morreu? 
Resposta intermediária: Muhammad Ali tinha 74 anos quando morreu. 
Pergunta de acompanhamento: Quantos anos Alan Turing tinha quando morreu? 
Resposta intermediária: Alan Turing tinha 41 anos quando morreu. 
Então a resposta final é: Muhammad Ali 
""", 
    }, 
    {"pergunta": "Quando nasceu o fundador do craigslist?", 
     "resposta": 
"""São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Quem foi o fundador do craigslist? 
Resposta intermediária: O craigslist foi fundado por Craig Newmark. 
Pergunta de acompanhamento: Quando nasceu Craig Newmark? 
Resposta intermediária: Craig Newmark nasceu em 6 de dezembro de 1952. 
Então a resposta final é: 6 de dezembro de 1952 
""", 
    }, 
    {"pergunta": "Quem foi o avô materno de George Washington?",
     "resposta": 
"""São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Quem foi a mãe de George Washington? 
Resposta intermediária: A mãe de George Washington foi Mary Ball Washington. 
Pergunta de acompanhamento: Quem foi o pai de Mary Ball Washington? 
Resposta intermediária: O pai de Mary Ball Washington foi Joseph Ball. 
Então a resposta final é: Joseph Ball 
""", 
    },
    {"pergunta": "Os diretores de Jaws e Casino Royale são do mesmo país?", 
     "resposta": 
"""São necessárias perguntas de acompanhamento aqui: Sim. 
Pergunta de acompanhamento: Quem é o diretor de Jaws? 
Resposta Intermediária: O diretor de Jaws é Steven Spielberg. 
Pergunta de acompanhamento: De onde é Steven Spielberg? 
Resposta Intermediária: Estados Unidos. 
Pergunta de acompanhamento: Quem é o diretor de Casino Royale? 
Resposta Intermediária: O diretor de Casino Royale é Martin Campbell. 
Pergunta de acompanhamento: De onde é Martin Campbell? 
Resposta Intermediária: Nova Zelândia. 
Então a resposta final é: Não 
""",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ('human', '{pergunta}'),
        ('ai', '{resposta}')
    ]
)

few_shot_template = FewShotChatMessagePromptTemplate(
    examples=exemplos,
    example_prompt=example_prompt,
)

prompt_tamplete = ChatPromptTemplate.from_messages(
    [
    few_shot_template,
    ('human','{input}')
    ]
)

prompt = prompt_tamplete.format_messages(input='Quem fez mais gols, romário ou pelé?')

resposta = chat.invoke(prompt)
print(resposta.content)