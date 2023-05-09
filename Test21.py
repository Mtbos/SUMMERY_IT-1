from string import punctuation
import re
import nltk
import pyttsx3
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

engine = pyttsx3.init()
s = input(str('Enter to get the summary of anything you are not understanding:'))
s.lower()
s = re.sub(r'http\S+', '', s)
s = re.sub('[^a-zA-Z0-9\s]', '', s)

token = nltk.word_tokenize(s)
# print(token)

tagged = nltk.pos_tag(token)
# print(tagged)
stopword = set(stopwords.words('english'))
words = [word for word in token if word.lower() not in stopword and word not in punctuation]

tag = nltk.pos_tag(words)
# print(tag)

lemma = WordNetLemmatizer()
lemmas = [lemma.lemmatize(word) for word in words]
s = ' '.join(lemmas)
vector = TfidfVectorizer()
tf_vid = vector.fit_transform(lemmas)
matrix = cosine_similarity(tf_vid)
nx_graph = nx.from_numpy_array(matrix)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i], s) for i, s in enumerate([s])), reverse=True)
summary = ' '.join([ranked_sentences[i][1] for i in range(min(3, len(ranked_sentences)))])
print(summary)
engine.say(summary)
engine.runAndWait()
