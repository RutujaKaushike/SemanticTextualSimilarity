# import spacy
# from spacy import displacy
#
# nlp = spacy.load("en_core_web_sm")
# text = "I love mangoes"
# doc = nlp(text)
# sentence_spans = list(doc.sents)
# displacy.serve(sentence_spans, style="dep")
#


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from src.part2 import tokenize_text

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
#
# sen1 = "I am on a bank"
# sen2 = "I am in a bank"
#
# words1 = word_tokenize(sen1)
# words2 = word_tokenize(sen2)
#
# filtered_sentence1 = [w for w in words1 if not w in stop_words]
#
# filtered_sentence2 = [w for w in words1 if not w in stop_words]
#
# filtered_sentence1 = []
# filtered_sentence2 = []
#
#
# for w in words1:
#     if w not in stop_words:
#         filtered_sentence1.append(w)
#
# for w in words2:
#     if w not in stop_words:
#         filtered_sentence2.append(w)
#
# lemma1 = []
# for s in filtered_sentence1:
#     lemma1.append(lemmatizer.lemmatize(s))
#
# lemma2 = []
# for s in filtered_sentence2:
#     lemma2.append(lemmatizer.lemmatize(s))
#
# set1 = set(lemma1)
# print("Lemma Filtered sentence set 1",set1)
# set2 = set(lemma2)
# print("Lemma Filtered sentence set 2",set2)
#
# len1 = len(set1)
# print(len1)
# len2 = len(set2)
# print(len2)
#
# len3 = len(set1.intersection(set2))
# print("Filtered sentence intersection ",set1.intersection(set2))
# print(len3)
#
# print(float(len3/(len1+len2-len3)))

text = "You are my sun and stars"

tokenized_data = tokenize_text(text)
pos_tag(tokenized_data)
print(wn.synsets('great', pos='n'))
