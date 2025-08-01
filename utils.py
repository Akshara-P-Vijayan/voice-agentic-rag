from gtts import gTTS
from tempfile import NamedTemporaryFile
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

def parse_pdf(file):
    doc = fitz.open("pdf", file.read())
    return " ".join([page.get_text() for page in doc])

def parse_html(file):
    soup = BeautifulSoup(file.read(), 'html.parser')
    return soup.get_text()

def speak_response(text):
    tts = gTTS(text=text, lang='en')
    with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return open(fp.name, 'rb').read()
