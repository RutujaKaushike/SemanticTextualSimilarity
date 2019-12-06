from nltk import word_tokenize, WordNetLemmatizer


def bag_of_words(sen1, sen2):
    sen1 = sen1.replace(".","")
    sen2 = sen2.replace(".","")

    lemmatizer = WordNetLemmatizer()
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

    print(word2count3)
    # print(word2count1)

    print(words1)
    vector1 = []
    for wc in word2count3:
        if wc in lemma1:
            vector1.append(word2count1[wc])
        else:
            vector1.append(0)

    print(vector1)

    print(words2)
    vector2 = []
    for wc in word2count3:
        if wc in lemma2:
            vector2.append(word2count2[wc])
        else:
            vector2.append(0)

    print(vector2)

    c = 0
    for i in range(len(vector1)):
        c += vector1[i]*vector2[i]
    cosine = c / float((sum(vector1)*sum(vector2))**0.5)
    print("similarity: ", cosine)


# sen1 = "John likes to watch movies. Mary likes movies too."
# sen2 = "John also likes to watch football games."
sen1 = "I love mangoes"
sen2 = "I love fruits"

bag_of_words(sen1, sen2)