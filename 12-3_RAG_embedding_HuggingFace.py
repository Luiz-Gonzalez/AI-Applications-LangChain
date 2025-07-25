from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

# sentence similarity
model = 'all-MiniLM-L6-v2'

embedding_model = HuggingFaceBgeEmbeddings(model_name = model)

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