import pandas as pd
from os import path
import csv
import random
import sys
from unicodedata import category
from math import ceil

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


if __name__ == "__main__":

    num_sentences = 10000
    n_sentences_per_language = ceil(num_sentences/10)
    n_sentences_per_accent = ceil(num_sentences/7)

    script_dir = path.dirname(__file__) + "/raw_corpora"

    # load english corpora ------------------------------------------------------------------------------------------
    en_path = "\english-corpora"

    en_au_path = script_dir + en_path + "\eng-au" + "_sentences.txt"
    en_ca_path = script_dir + en_path + "\eng-ca" + "_sentences.txt"
    en_dm_path = script_dir + en_path + "\eng-dm" + "_sentences.txt"
    en_eu_path = script_dir + en_path + "\eng-eu" + "_sentences.txt"
    en_nz_path = script_dir + en_path + "\eng-nz" + "_sentences.txt"
    en_uk_path = script_dir + en_path + "\eng-uk" + "_sentences.txt"
    en_za_path = script_dir + en_path + "\eng-za" + "_sentences.txt"

    en_au_list = pd.read_csv(en_au_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    en_ca_list = pd.read_csv(en_ca_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    en_dm_list = pd.read_csv(en_dm_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    en_eu_list = pd.read_csv(en_eu_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    en_nz_list = pd.read_csv(en_nz_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    en_uk_list = pd.read_csv(en_uk_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    en_za_list = pd.read_csv(en_za_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()

    random.shuffle(en_au_list)
    random.shuffle(en_ca_list)
    random.shuffle(en_dm_list)
    random.shuffle(en_eu_list)
    random.shuffle(en_nz_list)
    random.shuffle(en_uk_list)
    random.shuffle(en_za_list)

    en_sentence_list = en_au_list[0:n_sentences_per_accent] + en_ca_list[0:n_sentences_per_accent] + \
                    en_dm_list[0:n_sentences_per_accent] + en_eu_list[0:n_sentences_per_accent] + \
                    en_nz_list[0:n_sentences_per_accent] + en_uk_list[0:n_sentences_per_accent] + \
                    en_za_list[0:n_sentences_per_accent]

    random.shuffle(en_sentence_list)
    en_sentence_list = en_sentence_list[0: num_sentences]

    # load non-english corpora ------------------------------------------------------------------------------------------

    non_en_path = "/non-english-corpora"

    ces_path = script_dir + non_en_path + "/ces" + "_sentences.txt"
    deu_path = script_dir + non_en_path + "/deu" + "_sentences.txt"
    fra_path = script_dir + non_en_path + "/fra" + "_sentences.txt"
    ita_path = script_dir + non_en_path + "/ita" + "_sentences.txt"
    nor_path = script_dir + non_en_path + "/nor" + "_sentences.txt"
    pol_path = script_dir + non_en_path + "/pol" + "_sentences.txt"
    rus_path = script_dir + non_en_path + "/rus" + "_sentences.txt"
    spa_path = script_dir + non_en_path + "/spa" + "_sentences.txt"
    swe_path = script_dir + non_en_path + "/swe" + "_sentences.txt"
    tur_path = script_dir + non_en_path + "/tur" + "_sentences.txt"

    ces_list = pd.read_csv(ces_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    deu_list = pd.read_csv(deu_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    fra_list = pd.read_csv(fra_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    ita_list = pd.read_csv(ita_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    nor_list = pd.read_csv(nor_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    pol_list = pd.read_csv(pol_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    rus_list = pd.read_csv(rus_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    spa_list = pd.read_csv(spa_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    swe_list = pd.read_csv(swe_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()
    tur_list = pd.read_csv(tur_path, sep="\t", encoding="utf-8", header=None, quoting=csv.QUOTE_NONE )[1].tolist()

    random.shuffle(ces_list)
    random.shuffle(deu_list)
    random.shuffle(fra_list)
    random.shuffle(ita_list)
    random.shuffle(nor_list)
    random.shuffle(pol_list)
    random.shuffle(rus_list)
    random.shuffle(spa_list)
    random.shuffle(swe_list)
    random.shuffle(tur_list)

    non_en_sentence_list = ces_list[0:n_sentences_per_language] + deu_list[0:n_sentences_per_language] + \
                        fra_list[0:n_sentences_per_language] + ita_list[0:n_sentences_per_language] + \
                        nor_list[0:n_sentences_per_language] + pol_list[0:n_sentences_per_language] + \
                        rus_list[0:n_sentences_per_language] + spa_list[0:n_sentences_per_language] + \
                        swe_list[0:n_sentences_per_language] + tur_list[0:n_sentences_per_language]

    random.shuffle(non_en_sentence_list)
    non_en_sentence_list = non_en_sentence_list[0: num_sentences]

    chrs = (chr(i) for i in range(sys.maxunicode + 1))
    utf_8_letters = list(c for c in chrs if category(c).startswith("L"))

    clean_non_en_list = clean_sentance_list(non_en_sentence_list, utf_8_letters, excep=["'", " "])
    clean_en_list = clean_sentance_list(en_sentence_list, utf_8_letters, excep=["'", " "])

    output_dir = path.dirname(__file__) + "/filtered_corpora/"

    with open(output_dir + "english.txt", "w", encoding="utf-8") as english_output:
        for sentence in clean_en_list:
            english_output.write(sentence + '\n')

    with open(output_dir + "non_english.txt", "w", encoding="utf-8") as non_english_output:
        for sentence in clean_non_en_list:
            non_english_output.write(sentence + '\n')