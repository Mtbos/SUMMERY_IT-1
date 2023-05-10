import re
import string
import nltk
import pyttsx3
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

engine = pyttsx3.init()


def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'https?\S+', '', text)
    tokens = nltk.word_tokenize(text)
    stopword = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopword]
    tagged = nltk.pos_tag(tokens)
    le = WordNetLemmatizer()
    lemmas = [le.lemmatize(word) for word in tokens]
    return ' '.join(lemmas)


def generate_summary(text):
    preprocessed_text = preprocess(text)
    tokenizer = Tokenizer("english")
    parser = PlaintextParser.from_string(preprocessed_text, tokenizer)
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=1)
    return summary[0]


def generate_conclusion(text, summary):
    conclusion = "Based on the summary, it can be concluded that {}.".format(text.lower())
    return conclusion


txt = 'Welcome to The MT Creation. IM SUMMARYIT. VERSION-2. I am made to conclude large paragraphs in a small text. Please enter the text you want to summarize or conclude:'
engine.say(txt)
engine.runAndWait()

text = input(str('Enter the text below:'))
summary = generate_summary(text)
conclusion = generate_conclusion(text, summary)
# explanation = "The summary states that the fox is quick and hungry, while the dog is lazy and peaceful. Despite the fox being more agile, the dog is no match for it. As a result, the fox decides to find easier prey elsewhere. Based on this, it can be concluded that the fox is a skilled hunter that is always looking for an easy target."
# print("Summary:", summary)
print("Conclusion:", conclusion)
engine.say(conclusion)
engine.runAndWait()
# print("Explanation:", explanation)
