import fitz
from gtts import gTTS
from tempfile import NamedTemporaryFile
from IPython.display import Audio, display

def parse_pdf(file):
    doc = fitz.open("pdf", file.read())
    return " ".join(page.get_text() for page in doc)

def parse_html(file):
    from bs4 import BeautifulSoup
    return BeautifulSoup(file.read(), "html.parser").get_text()

def speak_response(text):
    tts = gTTS(text=text, lang="en")
    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        data = open(tmp.name, "rb").read()
    return data
