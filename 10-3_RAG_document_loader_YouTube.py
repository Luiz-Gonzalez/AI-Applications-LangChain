
# foi necessário instalar o pacote ffmpeg pelo terminal do mac: brew install ffmpeg

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

url = 'https://www.youtube.com/watch?v=d4EYFAvSkqo'

# onde será salvo o áudio do vídeo
save_dir='docs/youtube/'

# generic faz mais de uma coisa ao mesmo tempo
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    # audio para texto:
    OpenAIWhisperParser()
)
docs = loader.load()

print(docs[0].page_content)