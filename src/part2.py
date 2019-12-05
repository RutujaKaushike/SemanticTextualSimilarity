import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet

from spacy import displacy

def tokenize_text(text):
    return word_tokenize(text)

def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemma = []
    for w in words:
        lemma.append(lemmatizer.lemmatize(w))
    return lemma

def stop_word_remove_text(text):
    tokenized_data = tokenize_text(text)
    stop_word_data = set(stopwords.words('english'))
    words_without_stopwords = [w for w in tokenized_data if not w in stop_word_data]
    return words_without_stopwords

def partofspeech_tag(text):
    tokenized_data = tokenize_text(text)
    return pos_tag(tokenized_data)

def display_dependency_parse_tree(text):
    spacy_model = spacy.load("en_core_web_sm")
    text_to_be_shown = spacy_model(text)
    displacy.serve(text_to_be_shown, style="dep")

def hypernym_text(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    if(lemma):
        most_frequent_synset_of_lemma = wordnet.synsets(lemma)[0]
        print(most_frequent_synset_of_lemma.hypernyms())
    else:
        print("Please enter a valid word")

def hyponym_text(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    if(lemma):
        most_frequent_synset_of_lemma = wordnet.synsets(lemma)[0]
        print(most_frequent_synset_of_lemma.hyponyms())
    else:
        print("Please enter a valid word")

def meronym_text(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    if(lemma):
        most_frequent_synset_of_lemma = wordnet.synsets(lemma)[0]
        print("part meronyms are\n")
        print(most_frequent_synset_of_lemma.part_meronyms())  # for part meronyms
        print("\n substance meronyms are\n")
        print(most_frequent_synset_of_lemma.substance_meronyms()) # for substance meronyms
    else:
        print("Please enter a valid word")

def holonym_text(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    if(lemma):
        most_frequent_synset_of_lemma = wordnet.synsets(lemma)[0]
        print("part holonyms are\n")
        print(most_frequent_synset_of_lemma.part_holonyms())  # for part holonyms
        print("\n substance holonyms are\n")
        print(most_frequent_synset_of_lemma.substance_holonyms()) # for substance holonyms
    else:
        print("Please enter a valid word")

if __name__ == '__main__':
    print("\nPlease select operation - \n"\
          "1. Tokenize data \n"\
          "2. Lemmatize data \n"\
          "3. Remove stop words from data\n"\
          "4. Display part-of-speech (pos) tagging\n"\
          "5. Display dependency parse tree\n"\
          "6. Hypernym \n"\
          "7. Hyponym \n"\
          "8. Meronym \n"\
          "9. Holonym\n"\
          "10. Quit\n")

    select = input("Select operations form 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10:\n")

    while select != '10':
        if select == '1':
            text = input("Enter text:\n")
            print("Tokenized data is: \n")
            print(tokenize_text(text))
        elif select == '2':
            text = input("Enter text:\n")
            print("Lemmatized data is: \n")
            print(lemmatize_text(text))
        elif select == '3':
            text = input("Enter text:\n")
            print("Cleaned data after removing stop words is: \n")
            print(stop_word_remove_text(text))
        elif select == '4':
            text = input("Enter text:\n")
            print("POS tagged data is: \n")
            print(partofspeech_tag(text))
        elif select == '5':
            text = input("Enter text:\n")
            print("Displaying the dependency parse tree \n")
            display_dependency_parse_tree(text)
        elif select == '6':
            text = input("Enter a word:\n")
            print("Hypernyms are: \n")
            lemmas = lemmatize_text(text)
            hypernym_text(lemmas[0])
        elif select == '7':
            text = input("Enter a word:\n")
            print("Hyponyms are: \n")
            lemmas = lemmatize_text(text)
            hyponym_text(lemmas[0])
        elif select == '8':
            text = input("Enter a word:\n")
            print("Meronyms are: \n")
            lemmas = lemmatize_text(text)
            meronym_text(lemmas[0])
        elif select == '9':
            text = input("Enter a word:\n")
            print("Holonyms are: \n")
            lemmas = lemmatize_text(text)
            holonym_text(lemmas[0])
        else:
            print("Please select valid operation.")

        print("\nPlease select operation - \n" \
              "1. Tokenize data \n" \
              "2. Lemmatize data \n" \
              "3. Remove stop words from data\n" \
              "4. Display part-of-speech (pos) tagging\n" \
              "5. Display dependency parse tree\n" \
              "6. Hypernym \n"
              "7. Hyponym \n" \
              "8. Meronym \n" \
              "9. Holonym\n" \
              "10. Quit\n")

        select = input("Select operations form 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10:\n")

    print("Proceeding to exit")