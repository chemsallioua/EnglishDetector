import collections

from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk import ngrams
import random

def bag_of_words(words, feature ="word"):
    return [{feature: word} for word in words]

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)

def bag_of_non_stopwords(words, stopfile='and'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)

def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
            return label_feats
        
def split_label_feats(lfeats, split=0.75):
    cutoff = int(len(lfeats) * split)
    random.shuffle(lfeats)
    return lfeats[:cutoff], lfeats[cutoff:]

def bag_of_trigrams_char(words):
    return bag_of_words(ngrams_char(words), feature ="trigram")

def featuresets_from_words(words_list, labels, feature_detector=bag_of_words):
    featuresets = []
    for label in labels:
        for words in words_list:
            feats = feature_detector(words)
            featuresets =  featuresets + [(feat, label) for feat in feats]
    return featuresets

def document_word_features(document, word_features): 
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

def document_ngram_char_features(document, ngram_features, n, left_pad =  False, right_pad =  False): 
    document_ngrams = set(ngrams_char(document, n, left_pad = left_pad, right_pad = right_pad))
    features = {}
    for gram in ngram_features:
        features[gram] = (gram in document_ngrams)
    return features

def ngrams_char(words, n, left_pad = False, right_pad = False):
    ngrams_char = []
    for word in words:
        #print(word)
        n_grams = ngrams(word, n,
                         pad_left = left_pad, \
                         left_pad_symbol ="$", pad_right = right_pad, \
                         right_pad_symbol ="$") \

        for g in n_grams:
            gram = ""
            for i in range(0,len(g)):
                gram = gram + str(g[i]) 
            ngrams_char.append(gram)
    # print(trigrams_char)
    return ngrams_char