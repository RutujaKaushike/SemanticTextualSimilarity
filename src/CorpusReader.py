import pprint
import argparse
from collections import defaultdict

from googletrans import Translator
translator = Translator()

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--pairfile', type=str, required=True)

def main(args):
    with open(args.pairfile,'r') as f:
        sentences = []
        ids = []
        first_sents = []
        second_sents = []
        true_score = []
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
        pp.pprint(sentences)

    Counts_for_tf = defaultdict(int)

    for sent in first_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    pp.pprint(Counts_for_tf)
    for sent in second_sents:
        for w in [w.strip("?.,") for w in sent.split()]: Counts_for_tf[w] += 1
    pp.pprint(Counts_for_tf)

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)