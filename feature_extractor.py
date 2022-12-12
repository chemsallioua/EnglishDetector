from nltk import ngrams
import random
import sys
from unicodedata import category
        
def split_label_feats(lfeats, split=0.75):
    cutoff = int(len(lfeats) * split)
    random.shuffle(lfeats)
    return lfeats[:cutoff], lfeats[cutoff:]

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

def clean_sentance_list(sentence_list, stop_char=None ,excep=["'", " "]):
    clean_sentances_list = []
    for sentence in sentence_list:
        clean_sentances_list.append(clean_string(sentence,stop_char, excep))   
    return clean_sentances_list


# removes all symbols, punctuation, and leading and closing whitespaces, 
def clean_string(string, stop_char = None, excep = ["'", " "]):

    if(stop_char == None):
        chrs = (chr(i) for i in range(sys.maxunicode + 1))
        stop_char = list(c for c in chrs if category(c).startswith("L"))

    sc = stop_char + excep

    string = str(string)
    for ch in string:
        if ch not in sc:
            string = string.replace(ch, '')
    # print("sentence is: " + string.strip())
    return string.strip().lower()