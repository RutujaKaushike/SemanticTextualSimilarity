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
from nltk import pos_tag
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVR
from nltk import word_tokenize, pos_tag, WordNetLemmatizer, tag
from nltk.corpus import wordnet as wn
import numpy as np
import spacy
import math
from functools import reduce

nlp = spacy.load("en_core_web_lg")

translator = Translator()

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--pairfile', type=str, required=True)
parser.add_argument('--testfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'): return 'n'
    if tag.startswith('V'): return 'v'
    if tag.startswith('J'): return 'a'
    if tag.startswith('R'): return 'r'
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def sentence_similarity_word_alignment(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet and ppdb """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    ppdb_score, align_cnt = 0, 0
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        L = [synset.path_similarity(ss) for ss in synsets2]
        L_prime = L
        L = [l for l in L if l]

        # Check that the similarity could have been computed

        if L:
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count >0: score /= count

    return score

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

# def jaccard(s1, s2):
#     sentence1 = pos_tag(word_tokenize(s1))
#     sentence2 = pos_tag(word_tokenize(s2))
#
#     syn1 = set()
#     syn2 = set()
#
#     for synset in wn.synsets(sentence1):
#
#         for lemma in synset.lemmas():
#
#             syn1.append(lemma.name())
#
#     for synset in wn.synsets(sentence2):
#
#         for lemma in synset.lemmas():
#
#             syn2.append(lemma.name())
#
#     a = len(syn1)
#     b = len(syn2)
#     c = syn1.intersection(syn2)
#     d = len(c)
#     print(float(d/ (a+b-d)))
#     return float(d/ (a+b-d))
#


# def jaccard(s1, s2):
#     sentence1 = pos_tag(word_tokenize(s1))
#     sentence2 = pos_tag(word_tokenize(s2))
#
#     syn1 = set()
#     syn2 = set()
#
#     for synset in wn.synsets(sentence1):
#
#         for lemma in synset.lemmas():
#
#             syn1.append(lemma.name())
#
#     for synset in wn.synsets(sentence2):
#
#         for lemma in synset.lemmas():
#
#             syn2.append(lemma.name())
#
#     a = len(syn1)
#     b = len(syn2)
#     c = syn1.intersection(syn2)
#     d = len(c)
#     print(float(d/ (a+b-d)))
#     return float(d/ (a+b-d))
#


def jaccard_distance(txt1,txt2):
    v1=[]
    v2=[]
    if txt1.get("nsubj") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("nsubj"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("nsubj"))

    if txt1.get("dobj") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("dobj"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("dobj"))

    if txt1.get("pobj") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("pobj"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("pobj"))

    if txt1.get("root") is not None:
        pos_subj1 = tag.pos_tag([lemmatizer.lemmatize(txt1.get("root"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj1]
        for i in syn:
            if i is not None:
                v1.append(i)
            else:
                v1.append(txt1.get("root"))
    if txt2.get("nsubj") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("nsubj"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("nsubj"))
    if txt2.get("dobj") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("dobj"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("dobj"))

    if txt2.get("pobj") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("pobj"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("pobj"))

    if txt2.get("root") is not None:
        pos_subj2 = tag.pos_tag([lemmatizer.lemmatize(txt2.get("root"))])
        syn = [tagged_to_synset(*tagged_word) for tagged_word in pos_subj2]
        for i in syn:
            if i is not None:
                v2.append(i)
            else:
                v2.append(txt1.get("root"))
    # print(v1)
    # print(v2)
    intersection = set(v1).intersection(set(v2))
    union = set(v1).union(set(v2))
    jd =  len(intersection)/len(union)
    # print(jd)
    return jd

def main(args):
    with open(args.pairfile,'r',encoding="utf8") as f:
        sentences = []
        ids = []
        first_sents = []
        second_sents = []
        true_score = []
        next(f)
        for line in f.readlines():
            line_split = line.split('\t')
            id = line_split[0]
            first_sentence = line_split[1]
            second_sentence = line_split[2]
            gold_tag = line_split[3].strip('\n')
            ids.append(id)
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)
            true_score.append(gold_tag)
            sentences += [id, first_sentence, second_sentence, gold_tag]

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

        scores = [
                   sentence_similarity_word_alignment(s1,s2),
                    text_similarity(doc1, doc2),
            bag_of_words(s1, s2),
            parse_similarity(s1,s2),
            semanticSimilarity(s1,s2)
        ]

        # cosine similarity

        feature_scores.append(scores)
    # print(feature_scores)
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(feature_scores);
    X_features = scaler.transform(feature_scores)
    clf = RandomForestClassifier(n_estimators=100)

    # clf = BaggingRegressor(SVR(kernel='linear'), n_estimators=15) # R1 uses default parameters as described in SVR documentation
    # clf = SVR(kernel='linear')
    # clf.fit(X_features, true_score)
    model2 = AdaBoostClassifier(n_estimators=50,
                                learning_rate=1)
    # model2.fit(X_features, true_score)
    # model3 = SVR(kernel='linear')
    # model3.fit(X_features, true_score)
    # from sklearn.externals import joblib

    # Save the model as a pickle in a file
    # joblib.dump(model3, 'model3.pkl')


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
            first_sentence = line_split[1]
            second_sentence = line_split[2]
            ids.append(id)
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)

    for sent in first_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    for sent in second_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1

    feature_scores = []
    N = len(first_sents)
    for i in range(N):
        s1 = first_sents[i]
        s2 = second_sents[i]


        scores = [
                  sentence_similarity_word_alignment(s1,s2),
                  text_similarity(nlp(s1),nlp(s2)),
            bag_of_words(s1, s2),
            parse_similarity(s1,s2),
            semanticSimilarity(s1,s2)
            ]
        feature_scores.append(scores)

    X_features = scaler.transform(feature_scores)
    # Y_pred_np = clf.predict(X_features)
    # Y_model2 = model2.predict(X_features)
    # Y_model3 = model3.predict(X_features)

    Y_eclf = eclf.predict(X_features)

    # Y_pred_np = [min(5,max(0,p),p) for p in int(Y_pred_np)]
    with open(args.predfile,'w') as f_pred:
        f_pred.write('Input_Id \t Predicted_Tage\n')
        for i in range(len(Y_eclf)):
            f_pred.write(ids[i]+"\t"+str(Y_eclf[i])+'\n')

def text_similarity(inp_obj1, inp_obj2):
    return inp_obj1.similarity(inp_obj2)

def chunk_similarity(inp_obj1, inp_obj2):
    noun = []
    s1len = 0
    s2len = 0

    if(inp_obj1 and inp_obj1.vector_norm):
        for chunk1 in inp_obj1.noun_chunks:
            s1len +=1
            if(inp_obj2 and inp_obj2.vector_norm):
                for chunk2 in inp_obj2.noun_chunks:
                    s2len +=1
                    sim = chunk1.similarity(chunk2)
                    if(sim!=None or sim):
                        sim =  0.0
                    if(math.isnan(sim)):
                        sim=0.0
                    noun.append(sim)
    minlen = min(s1len,s2len)
    noun = (sorted(noun, reverse=True)[:minlen])

    return np.mean(noun)

def token_similarity(doc):
    for token1 in doc:
        for token2 in doc:
            return token1.similarity(token2)


def Average(lst):
    average = 0
    sum = 0
    for num in lst:
        sum = sum+num;
    if len(lst) is not 0:
        average = sum / len(lst)
    return round(average)

def parse_similarity(s1,s2):
    s1 = s1.translate(str.maketrans('', '', string.punctuation))
    s2 = s2.translate(str.maketrans('', '', string.punctuation))
    # s2 = s2.replace(".","")
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


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)







def sentence_similarity_simple_baseline(s1, s2,counts = None):
    def embedding_count(s):
        ret_embedding = defaultdict(int)
        for w in s.split():
            w = w.strip('?.,')
            ret_embedding[w] += 1
        return ret_embedding
    first_sent_embedding = embedding_count(s1)
    second_sent_embedding = embedding_count(s2)
    Embedding1 = []
    Embedding2 = []
    if counts:
        for w in first_sent_embedding:
            Embedding1.append(first_sent_embedding[w] * 1.0/ (counts[w]+0.001))
            Embedding2.append(second_sent_embedding[w] *1.0/ (counts[w]+0.001))
    else:
        for w in first_sent_embedding:
            Embedding1.append(first_sent_embedding[w])
            Embedding2.append(second_sent_embedding[w])
    ret_score = 0
    if not 0 == sum(Embedding2):
        #https://stackoverflow.com/questions/6709693/calculating-the-similarity-of-two-lists
        # https://docs.python.org/3/library/difflib.html
        sm= difflib.SequenceMatcher(None,Embedding1,Embedding2)
        ret_score = sm.ratio()*5
    return ret_score

brown_ic = wordnet_ic.ic('ic-brown.dat')

def sentence_similarity_information_content(sentence1, sentence2):

    ''' compute the sentence similairty using information content from wordnet '''
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    ppdb_score, align_cnt = 0, 0
    # For each word in the first sentence
    for synset in synsets1:
        L = []
        for ss in synsets2:
            try:
                L.append(synset.res_similarity(ss, brown_ic))
            except:
                continue
        if L:
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count >0: score /= count
    return score

def extract_absolute_difference(s1, s2):
    """t \in {all tokens, adjectives, adverbs, nouns, and verbs}"""
    s1, s2 = word_tokenize(s1), word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = abs(len(s1) - len(s2)) / float(len(s1) + len(s2))
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return [t1, t2, t3, t4, t5]

def extract_overlap_pen(s1, s2):
    """
    :param s1:
    :param s2:
    :return: overlap_pen score
    """
    ss1 = s1.strip().split()
    ss2 = s2.strip().split()
    ovlp_cnt = 0
    for w1 in ss1:
        ovlp_cnt += ss2.count(w1)
    score = 2 * ovlp_cnt / (len(ss1) + len(ss2) + .0)
    return score