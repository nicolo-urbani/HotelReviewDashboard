import string
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
from nltk.tokenize import word_tokenize

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # drop spaces start and end text
    text = text.strip()
    # tokenize text and remove punctuation
    text = word_tokenize(text)
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = set(stopwords.words('english'))
    stop = set([item for item in stop if item not in ["not", "nor", "against"]])
    additional_stopwords = [
        "The", "And", "I", "J", "K", "I'd", "That's", "\x81", "It", "I'm", "...", "\x89", "ĚĄ",
        "it's", "ă", "\x9d", "âÂĺ", "Ě", "˘", "Â", "âÂ", "Ň", "http", "https", "co", "000",
        "Ň", "Ň", "Ň", "ââ", 'ě', 'ň', "not",
        'didnt', 'did', 'havent', 'week', 'hi', 'wa', 'ha', 'day', 'today', 'really', 'also',
        'go', 'us', 'dont', 'got', 'im', 'ive', 'burger', 'food', 'came', 'back',
        'get', 'try', 'would', 'time', 'good', 'great', 'service', 'didn', 'definitely', 'hotel', 'went', 'took', 'left',
        'check', 'told', 'asked', 'like', 'don', 'wasn', 'hotels', 'just', 'don', 'said', 'people', 've', 'stay', 'stayed',
        'loved', 'com', 'night', 'birthday', 'free', 'touch', 'little', 'given', 'making', 'hear', 'recommend', 'card', 'make',
        'feel', 'days', 'differ', 'thier', 'couldn', 'breakfast', 'got', 'everyone', 'anything', 'everything', 'one', 'nothing', 'much'
    ]
    stop.update(additional_stopwords)
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return text

