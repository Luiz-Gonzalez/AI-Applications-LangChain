from langchain_community.document_loaders.web_base import WebBaseLoader

url = 'https://doc.clickup.com/9007047819/d/h/8cdt94b-4173/118fb96400a7b07/8cdt94b-67613'
loader = WebBaseLoader(url)
documentos = loader.load()

# ver o conteúdo page_content[caracter inicio:fim]
print(documentos[0].page_content[:1000])