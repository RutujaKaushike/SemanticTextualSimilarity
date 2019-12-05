import pprint
import argparse
import sklearn
from collections import defaultdict

from googletrans import Translator
from nltk import pos_tag
from sklearn.svm import SVR
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
translator = Translator()

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--pairfile', type=str, required=True)
parser.add_argument('--testfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)

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


        scores = [ sentence_similarity_word_alignment(s1,s2)]

        # cosine similarity
        feature_scores.append(scores)

    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(feature_scores);
    X_features = scaler.transform(feature_scores)
    #clf = LinearRegression(); clf.fit(X_features, true_score)
    # clf = BaggingRegressor(SVR(kernel='linear'), n_estimators=15) # R1 uses default parameters as described in SVR documentation
    clf = SVR(kernel='linear')
    clf.fit(X_features, true_score)

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

        scores = [sentence_similarity_word_alignment(s1,s2)]

        feature_scores.append(scores)

    X_features = scaler.transform(feature_scores)
    Y_pred_np = clf.predict(X_features)
    Y_pred_np = [min(5,max(0,p),p) for p in Y_pred_np]
    with open(args.predfile,'w') as f_pred:
        for i in range(len(Y_pred_np)):
            f_pred.write(str(Y_pred_np[i])+'\n')


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)