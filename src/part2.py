import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

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

if __name__ == '__main__':
    print("\nPlease select operation - \n"\
          "1. Tokenize data \n"\
          "2. Lemmatize data \n"\
          "3. Remove stop words from data\n"\
          "4. Display part-of-speech (pos) tagging\n"\
          "5. Display dependency parse tree\n"\
          "6. Quit\n")

    select = input("Select operations form 1, 2, 3, 4, 5 or 6:\n")

    while select != '6':
        text = input("Enter text:\n")

        if select == '1':
            print("Tokenized data is: \n")
            print(tokenize_text(text))
        elif select == '2':
            print("Lemmatized data is: \n")
            print(lemmatize_text(text))
        elif select == '3':
            print("Cleaned data after removing stop words is: \n")
            print(stop_word_remove_text(text))
        elif select == '4':
            print("POS tagged data is: \n")
            print(partofspeech_tag(text))
        elif select == '5':
            print("Displaying the dependency parse tree \n")
            display_dependency_parse_tree(text)
        else:
            print("Please select valid operation.")

        print("\nPlease select operation - \n" \
              "1. Tokenize data \n" \
              "2. Lemmatize data \n" \
              "3. Remove stop words from data\n" \
              "4. Display part-of-speech (pos) tagging\n" \
              "5. Display dependency parse tree\n" \
              "6. Quit\n")
        select = input("Select operations form 1, 2, 3, 4, 5 or 6:\n")

    print("Proceeding to exit")