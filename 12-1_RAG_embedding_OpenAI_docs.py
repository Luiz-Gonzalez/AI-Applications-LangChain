##############################
#
# RAG - Embedding - Docs
#
##############################

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

embeddings = embedding_model.embed_documents(
    [
        'Eu gosto de cachorros',
        'Eu gosto de animais',
        'O tempo está ruim lá fora',
    ]
)

# tamanho de cada vetor e valores max. e min.
for emb in embeddings:
    print(len(emb), max(emb), min(emb))

# comparação semântica - multiplicação entre vetores
# vetores mais próximos: o maior valor entre a multiplicacao dos vetores
print(np.dot(embeddings[0], embeddings[1]))
print(np.dot(embeddings[0], embeddings[2]))
print(np.dot(embeddings[1], embeddings[2]))