import difflib
import pprint
import argparse
import string
import re

from gensim.parsing.preprocessing import stem

import sklearn
from collections import defaultdict
from nltk.corpus import wordnet_ic
from nltk.metrics import edit_distance
from nltk.corpus import stopwords

from googletrans import Translator

from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVR
from nltk import word_tokenize, pos_tag, WordNetLemmatizer, tag
from nltk.corpus import wordnet as wn
import numpy as np
import spacy
import math

nlp = spacy.load("en_core_web_lg")

translator = Translator()

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--pairfile', type=str, required=True)
parser.add_argument('--testfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)

lemmatizer = WordNetLemmatizer()


# Feature
# This feature finds the similarity between spacy objects of two sentences
def text_similarity(inp_obj1, inp_obj2):
    return inp_obj1.similarity(inp_obj2)

# Feature
def bag_of_words(sen1, sen2):
    sen1 = sen1.replace(".","")
    sen2 = sen2.replace(".","")


    words1 = word_tokenize(sen1)
    lemma1 = []
    for w in words1:
        lemma1.append(lemmatizer.lemmatize(w))
    words2 = word_tokenize(sen2)
    lemma2 = []
    for w in words2:
        lemma2.append(lemmatizer.lemmatize(w))

    word2count1 = {}
    for data in lemma1:
        words = word_tokenize(data)
        for word in words:
            if word not in word2count1.keys():
                word2count1[word] = 1
            else:
                word2count1[word] += 1
    word2count2 = {}
    for data in lemma2:
        words = word_tokenize(data)
        for word in words:
            if word not in word2count2.keys():
                word2count2[word] = 1
            else:
                word2count2[word] += 1

    word2count3 = word2count1.copy()

    for word in word2count2.keys():
        if word in word2count1.keys():
            word2count3[word] += word2count2[word]
        else:
            word2count3[word] = word2count2[word]

    # vector1 = []
    # for i in range(len(lemma1)):
    #     vector1.append(0)
    #
    # vector2 = []
    # for i in range(len(lemma2)):
    #     vector2.append(0)

    # print(word2count3)
    # print(word2count1)

    # print(words1)
    vector1 = []
    for wc in word2count3:
        if wc in lemma1:
            vector1.append(word2count1[wc])
        else:
            vector1.append(0)

    # print(vector1)

    # print(words2)
    vector2 = []
    for wc in word2count3:
        if wc in lemma2:
            vector2.append(word2count2[wc])
        else:
            vector2.append(0)

    # print(vector2)

    c = 0
    for i in range(len(vector1)):
        c += vector1[i]*vector2[i]
    cosine = c / float((sum(vector1)*sum(vector2))**0.5)
    # print("similarity: ", cosine)
    return cosine

# This method calculates the Jaccard similarity between synsets of lemmatized forms of
# 'root', 'nsubj', 'pobj' and 'dobj'
def jaccard_distance(txt1,txt2):
    v1=[]
    v2=[]
    if txt1.get("nsubj") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("nsubj"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("nsubj"))

    if txt1.get("dobj") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("dobj"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("dobj"))

    if txt1.get("pobj") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("pobj"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("pobj"))

    if txt1.get("root") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("root"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("root"))
    if txt2.get("nsubj") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("nsubj"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("nsubj"))
    if txt2.get("dobj") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("dobj"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("dobj"))

    if txt2.get("pobj") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("pobj"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("pobj"))

    if txt2.get("root") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("root"))])
        syn = [taggedword_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("root"))

    intersection = set(v1).intersection(set(v2))
    union = set(v1).union(set(v2))
    jd =  len(intersection) / len(union)
    return jd

# Feature
# This feature extracts 'dobj', 'root', 'pobj' and 'nsubj' from the parse tree and then returns jaccard similarity score
def parse_similarity(s1,s2):
    s1 = s1.translate(str.maketrans('', '', string.punctuation))
    s2 = s2.translate(str.maketrans('', '', string.punctuation))
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    txt1 = {}
    txt2={}
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    tokens1 = [token.orth_ for token in tokenizer(s1)]
    tokens2 = [token.orth_ for token in tokenizer(s2)]
    i=-1
    for token in doc1:
        if(token.dep_=="nsubj"):
            txt1["nsubj"] = tokens1[i]
        elif(token.dep_=="dobj"):
            txt1["dobj"] = tokens1[i]
        elif(token.dep_=="pobj"):
            txt1["pobj"] = tokens1[i]
        elif(token.dep_=="ROOT"):
            txt1["root"] = tokens1[i]
    i=-1
    for token in doc2:
        i +=1
        if(token.dep_=="nsubj"):
            txt2["nsubj"] = tokens2[i]
        elif(token.dep_=="dobj"):
            txt2["dobj"] = tokens2[i]
        elif(token.dep_=="pobj"):
            txt2["pobj"] = tokens2[i]
        elif(token.dep_=="ROOT"):
            txt2["root"] = tokens2[i]
    return jaccard_distance(txt1, txt2)


def penntag_to_wntag(penntag):
    # This method converts Penn Treebank tag into Wordnet tag
    if penntag.startswith('N'):
        return 'n'
    if penntag.startswith('V'):
        return 'v'
    if penntag.startswith('J'):
        return 'a'
    if penntag.startswith('R'):
        return 'r'
    return None

def taggedword_to_synset(word, penntag):
    # This method returns synset of tagged word
    wordnetTag = penntag_to_wntag(penntag)
    if wordnetTag is None:
        return None
    try:
        return wn.synsets(word, wordnetTag)[0]
    except:
        return None

# Feature
# This feature finds out the similarity between the synsets of the tokenized words in the sentence.
# It finds the similarity between the most similar words in the sentences
def synset_similarity(s1, s2):
    s1 = pos_tag(word_tokenize(s1))
    s2 = pos_tag(word_tokenize(s2))
    syn1 = [taggedword_to_synset(*tagged_word) for tagged_word in s1]
    syn2 = [taggedword_to_synset(*tagged_word) for tagged_word in s2]
    # Removing None values
    syn1 = [sentencesimilarity for sentencesimilarity in syn1 if sentencesimilarity]
    syn2 = [sentencesimilarity for sentencesimilarity in syn2 if sentencesimilarity]
    score, count = 0.0, 0
    for syn in syn1:
        L = [syn.path_similarity(ss) for ss in syn2]
        L = [l for l in L if l]
        if L:
            best_score = max(L)
            score += best_score
            count += 1
    # Calculating the average of the values
    if count>0: score /= count
    return score




def path(set1, set2):
    return wn.path_similarity(set1, set2)


def wup(set1, set2):
    return wn.wup_similarity(set1, set2)


def edit(word1, word2):
    if float(edit_distance(word1, word2)) == 0.0:
        return 0.0
    return 1.0 / float(edit_distance(word1, word2))
STOP_WORDS = stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence

def semanticSimilarity(q1, q2):

    tokens_q1, tokens_q2 = tokenize(q1, q2)
    # stem_q1, stem_q2 = stemmer(tokens_q1, tokens_q2)
    tag_q1, tag_q2 = posTag(tokens_q1, tokens_q2)

    sentence = []
    for i, word in enumerate(tag_q1):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense1 = Lesk(sentence)
    sentence1Means = []
    for word in sentence:
        sentence1Means.append(sense1.lesk(word, sentence))

    sentence = []
    for i, word in enumerate(tag_q2):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense2 = Lesk(sentence)
    sentence2Means = []
    for word in sentence:
        sentence2Means.append(sense2.lesk(word, sentence))
    # for i, word in enumerate(sentence1Means):
    #     print sentence1Means[i][0], sentence2Means[i][0]

    R1 = computePath(sentence1Means, sentence2Means)
    R2 = computeWup(sentence1Means, sentence2Means)

    R = (R1 + R2) / 2

    # print R

    return overallSim(sentence1Means, sentence2Means, R)

def overallSim(q1, q2, R):

    sum_X = 0.0
    sum_Y = 0.0

    for i in range(len(q1)):
        max_i = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_i:
                max_i = R[i, j]
        sum_X += max_i

    for i in range(len(q1)):
        max_j = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_j:
                max_j = R[i, j]
        sum_Y += max_j

    if (float(len(q1)) + float(len(q2))) == 0.0:
        return 0.0

    overall = (sum_X + sum_Y) / (2 * (float(len(q1)) + float(len(q2))))

    return overall

def computeWup(q1, q2):

    R = np.zeros((len(q1), len(q2)))

    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = wup(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            if sim == None:
                sim = edit(q1[i][0], q2[j][0])

            R[i, j] = sim

    # print R

    return R

def computePath(q1, q2):

    R = np.zeros((len(q1), len(q2)))

    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = path(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            if sim == None:
                sim = edit(q1[i][0], q2[j][0])

            R[i, j] = sim

    # print R

    return R

def tokenize(q1, q2):
    """
        q1 and q2 are sentences/questions. Function returns a list of tokens for both.
    """
    return word_tokenize(q1), word_tokenize(q2)


def posTag(q1, q2):
    """
        q1 and q2 are lists. Function returns a list of POS tagged tokens for both.
    """
    return pos_tag(q1), pos_tag(q2)


def stemmer(tag_q1, tag_q2):
    """
        tag_q = tagged lists. Function returns a stemmed list.
    """

    stem_q1 = []
    stem_q2 = []

    for token in tag_q1:
        stem_q1.append(stem(token))

    for token in tag_q2:
        stem_q2.append(stem(token))

    return stem_q1, stem_q2

class Lesk(object):

    def __init__(self, sentence):
        self.sentence = sentence
        self.meanings = {}
        for word in sentence:
            self.meanings[word] = ''

    def getSenses(self, word):
        # print word
        return wn.synsets(word.lower())

    def getGloss(self, senses):

        gloss = {}

        for sense in senses:
            gloss[sense.name()] = []

        for sense in senses:
            gloss[sense.name()] += word_tokenize(sense.definition())

        return gloss

    def getAll(self, word):
        senses = self.getSenses(word)

        if senses == []:
            return {word.lower(): senses}

        return self.getGloss(senses)

    def Score(self, set1, set2):
        # Base
        overlap = 0

        # Step
        for word in set1:
            if word in set2:
                overlap += 1

        return overlap

    def overlapScore(self, word1, word2):

        gloss_set1 = self.getAll(word1)
        if self.meanings[word2] == '':
            gloss_set2 = self.getAll(word2)
        else:
            # print 'here'
            gloss_set2 = self.getGloss([wn.synset(self.meanings[word2])])

        # print gloss_set2

        score = {}
        for i in gloss_set1.keys():
            score[i] = 0
            for j in gloss_set2.keys():
                score[i] += self.Score(gloss_set1[i], gloss_set2[j])

        bestSense = None
        max_score = 0
        for i in gloss_set1.keys():
            if score[i] > max_score:
                max_score = score[i]
                bestSense = i

        return bestSense, max_score

    def lesk(self, word, sentence):
        maxOverlap = 0
        context = sentence
        word_sense = []
        meaning = {}

        senses = self.getSenses(word)

        for sense in senses:
            meaning[sense.name()] = 0

        for word_context in context:
            if not word == word_context:
                score = self.overlapScore(word, word_context)
                if score[0] == None:
                    continue
                meaning[score[0]] += score[1]

        if senses == []:
            return word, None, None

        self.meanings[word] = max(meaning.keys(), key=lambda x: meaning[x])

        return word, self.meanings[word], wn.synset(self.meanings[word]).definition()




def main(args):
    # Reading the train file
    with open(args.pairfile,'r') as f:
        sentences = []
        ids = []
        first_sents = []
        second_sents = []
        true_score = []
        next(f) #Skipping the first header line in the input file
        for line in f.readlines():
            line_split = line.split('\t')
            id = line_split[0]
            first_sent = line_split[1]
            second_sent = line_split[2]
            gold_tag = line_split[3].strip('\n')
            ids.append(id)
            first_sents.append(first_sent)
            second_sents.append(second_sent)
            true_score.append(gold_tag)
            sentences += [id, first_sent, second_sent, gold_tag]

    Counts_for_tf = defaultdict(int)
    for sent in first_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    for sent in second_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1

    feature_scores = []
    N = len(first_sents)


    for i in range(N):
        s1 = first_sents[i]
        s2 = second_sents[i]
        doc1 = nlp(s1)
        doc2 = nlp(s2)
        scores = [text_similarity(doc1, doc2), bag_of_words(s1, s2), parse_similarity(s1,s2),
                semanticSimilarity(s1,s2),synset_similarity(s1, s2)
        ]
        feature_scores.append(scores)

    scaler = sklearn.preprocessing.StandardScaler();
    scaler.fit(feature_scores);
    X_features = scaler.transform(feature_scores)
    # Model 1: Random Forest
    clf = RandomForestClassifier(n_estimators=100)
    # Model 2: Adaboost
    model2 = AdaBoostClassifier(n_estimators=50,
                                learning_rate=1)
    # Ensemble Model : Voting Classifier
    eclf = VotingClassifier(estimators=[('Random Forest', clf), ('Adaboost', model2)],
                            voting ='soft')
    eclf.fit(X_features, true_score)

    # Testing
    first_sents = []
    second_sents = []
    ids = []
    with open(args.testfile,'r',encoding="utf8") as f:
        next(f)
        for line in f.readlines():
            line_split = line.split('\t')
            id = line_split[0]
            first_sent = line_split[1]
            second_sent = line_split[2]
            ids.append(id)
            first_sents.append(first_sent)
            second_sents.append(second_sent)

    for sent in first_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    for sent in second_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1

    feature_scores = []
    N = len(first_sents)
    for i in range(N):
        s1 = first_sents[i]
        s2 = second_sents[i]
        scores = [text_similarity(doc1, doc2), bag_of_words(s1, s2), parse_similarity(s1,s2),
                  semanticSimilarity(s1,s2),synset_similarity(s1, s2)
                  ]
        feature_scores.append(scores)

    X_features = scaler.transform(feature_scores)
    Y_eclf = eclf.predict(X_features)
    with open(args.predfile,'w') as f_pred:
        f_pred.write('Input_Id \t Predicted_Tag\n')
        for i in range(len(Y_eclf)):
            f_pred.write(ids[i]+"\t"+str(Y_eclf[i])+'\n')




if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)

