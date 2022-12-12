print("Importing modules....")

from nltk.classify import NaiveBayesClassifier, accuracy
from nltk import FreqDist, LaplaceProbDist
from nltk.stem import WordNetLemmatizer
import random

from corpus_filtering import clean_string

from os import path

from feature_extractor import split_label_feats, ngrams_char, document_ngram_char_features, document_word_features

#nb_classifier = NaiveBayesClassifier.train(train_feats)

#en_bag_of_trigrams = bag_of_trigrams_char("meriem  sardou is  a beautiful  girl".split())

print("\nExtracting data from files....")


corpora_dir = path.dirname(__file__) + "/filtered_corpora/"

all_en_words = []
labeled_en_sentences_documents = []
labeled_non_en_sentences_documents = []
labeled_words_document = []
en_train_words = []
non_en_train_words = []

# sentence_legth = 4
# n_sentences = 10000
# n_grams = 3
# n_mostfreq_ngrams = 50
# n_mostfreq_words = 4000

sentence_legth = 4
n_sentences = 10000
n_grams = 3
n_mostfreq_ngrams = 50
n_mostfreq_words = 4000

lemmatizer = WordNetLemmatizer()
i = 0
with open(corpora_dir + 'english.txt', "r", encoding="utf-8") as f:
    for sentance in f:
        sen = sentance.split()
        lemmas = [lemmatizer.lemmatize(word, "v") for word in sen ]
        all_en_words.extend(lemmas)
        if (i < n_sentences):
            # print(sen[0:sentence_legth])
            # print(lemmas[0:sentence_legth])
            labeled_en_sentences_documents.append((lemmas[0:sentence_legth], "en"))
            en_train_words.extend(lemmas[0:sentence_legth])
            i = i +1

all_non_en_words = []
i = 0
with open(corpora_dir + 'non_english.txt', "r", encoding="utf-8") as f:
    for sentance in f:
        sen = sentance.split()
        all_non_en_words.extend(sen)
        if (i < n_sentences):
            labeled_non_en_sentences_documents.append((sen[0:sentence_legth], "non_en"))
            non_en_train_words.extend(sen[0:sentence_legth])
            i = i+1

labeled_sentences_documents = labeled_non_en_sentences_documents + labeled_en_sentences_documents
random.shuffle(labeled_sentences_documents)
# print(labeled_sentences_documents)

# labeled_words_document = [(word, "en") for word in all_en_words] + [(word, "non_en") for word in all_non_en_words]
# random.shuffle(labeled_words_document)

print("This is a snap on how your data looks:.....")
for i in range(0,10):
    print(labeled_sentences_documents[i])
print("....")
print("Data extracted and labeled successfully!", end = "\n\n")

print("You have:", end = "\n\n")

print("English words count : ", len(en_train_words))
print("Non-English words count : ", len(non_en_train_words), end="\n\n")

print("English senstences count : ", len(labeled_en_sentences_documents))
print("Non-English senstences count : ", len(labeled_non_en_sentences_documents), end="\n\n")

print("-------------------------------------------------------------------------")

print("Exctracting the most frequent ngrams...")
all_en_ngrams = FreqDist(gram for gram in ngrams_char(all_en_words, n_grams))
en_ngram_features = list(all_en_ngrams)[:n_mostfreq_ngrams]
en_words_features = list(FreqDist(all_en_words))[:n_mostfreq_words]
print("ten most frequent ngrams: ", en_ngram_features[1:11])
print("ten most frequent words: ", en_words_features[1:11] , end= "\n\n")

print("Forming featuresets....." )
# featuresets = [(document_ngram_char_features(d, en_ngram_features, n_grams), c) for (d,c) in labeled_words_document]
# random.shuffle(featuresets)
featuresets = [(document_ngram_char_features(d, en_ngram_features, n_grams) | document_word_features(d, en_words_features), c) for (d,c) in labeled_sentences_documents]
random.shuffle(featuresets)

# print("This is how your featuresets looks:.....\n")
# print("Number of en ngrams: ")

# print("...\n...")

train_set, test_set = split_label_feats(featuresets, split=0.75)
print("DONE.....", end= "\n\n" )
print("-------------------------------------------------------------------------")

print("Now traing our Naive Bayes Classifier to detect ngrams in words" )
classifier = NaiveBayesClassifier.train(train_set,LaplaceProbDist)

print("FINISHED! This is the accuracy of the model: ",accuracy(classifier, test_set))
classifier.show_most_informative_features(5)
print("\n")

while(True):
    print("-------------------------------------------------------------------------")
    user_sentence = input("Now enter a word that you want to classify: ")
    user_sentence = [lemmatizer.lemmatize(word, "v") for word in clean_string(user_sentence).split() ]
    print("Thank you! This is your word: ", user_sentence, end = "\n\n")
    feature = document_ngram_char_features(user_sentence, en_ngram_features, n_grams) | document_word_features(user_sentence, en_words_features)
    print("English feautures:\n ", [i for i in feature if feature[i]==True], "\n")
    # print([ft for ft in feature if ft.values() == True])
    print("The text you entered is in |" , classifier.classify(feature), "language |")
    print("P(en|s): ",classifier.prob_classify(feature).prob("en"),"\nP(non_en|s): ", classifier.prob_classify(feature).prob("non_en"), "\n")



