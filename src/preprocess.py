import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


nltk.download('stopwords')
def download_nltk():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

download_nltk()
stop_words = set(stopwords.words('english'))

# Keep only safe domain stopwords
custom_stopwords = ['resume', 'curriculum', 'cv']
stop_words.update(custom_stopwords)

lemmatizer = WordNetLemmatizer()



def clean_text(text: str) -> str:
    text = str(text).lower()

    # Normalize spacing
    text = re.sub(r'\s+', ' ', text)

    # Remove emails and URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)

    # Keep numbers + tech symbols
    text = re.sub(r'[^a-zA-Z0-9+.# ]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and len(w) > 1
    ]

    return " ".join(words)
