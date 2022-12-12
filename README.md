# Naive Bayes Based NLP model for English Language Identification with NLTK

The project's purpuse is to develop a small NLP classifier which classifies text as "EN" english or "NON-EN" non english language, by means of Python and the NLTK module. To do this, data scientists typically use neural network models. In this post, I demonstrate how to use a Naive Bayes model to build a straightforward English Language detection model.

**FOR MORE INFO ABOUT THE MODEL CHECK THE KAGGLE NOTEBOOK IN:**

https://www.kaggle.com/code/chemseddineallioua/naive-bayes-based-language-identification-system
## The Data
In order to train our NB model data from the Leipzig Corpora Collection have been used. check: https://wortschatz.uni-leipzig.de/en/download.
The corpora have the same format and are similar in size and content. They comprise randomly picked sentences in the corpus's language and range in size from 10,000 to 1 million sentences. The sources are either newspaper articles or materials gathered at random from the internet.

**The dataset is organized this way:**

- English Corpora:

    Seven ".txt" files containing 10k sentences each with different lengths. each ".txt" file contains different english accents. This gives us 70k sentences in english (included english accents: AU, CA, DM, EU, NZ, UK, ZA).

- Non-English Corpora:

    Ten ".txt" files containing 10k sentences each with different lengths. each ".txt" file contains a different language. The selected languages are only the languages which are based entirely or partially on latin letters. This gives us 100k sentences in a language different from english (included languages: CES, DEU, FRA, ITA, NOR, POL, RUS, SPA, SWE, TUR).

By means of a python script, the corpora have been cleaned (remove punctuation and other symbols) and packed in two ".txt" file with 10k sentences each for EN and NON-EN sentences.

## Preparing The Enviroment

The Machine Learning Model is implemented by means of NLTK version = 3.7 on Python 3.10.5.

To install NLTK:

    pip install nltk
The scripts make use also of numpy:

    pip install numpy

Also since the we make use of nltk.lemmitizer, it is important to donwload the nltk data package "omw-1.4" :

    >>import nltk
    >>nltk.download('omw-1.4')

## Running and training the classifier

In order to run the program, just run the _naive_bayes_predictor.py_ file:

    python .\naive_bayes_predictor.py

## Data pre-processing

The script _corpus_filtering.py_ served as data pre-processing module for the raw data downloaded from the **Leipzig Corpora Collection** (in the directory _/raw_corpora/_) to produce the filtred corpora in the _/filtered_corpora/_ directory.

It is not used during the normal running of the program.