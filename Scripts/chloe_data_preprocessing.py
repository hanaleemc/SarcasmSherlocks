import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sarc_data = pd.read_csv("train.En.csv")
sarc_data


import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
import string

df1 = sarc_data
pd.set_option('display.max_colwidth', None)

df = df1.drop(["Unnamed: 0", "rephrase", "sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"] , axis=1)
pd.set_option('display.max_colwidth', None)

df_clean = df
#lowercase
df_clean.tweet = df_clean.tweet.str.lower()
# @ user
df_clean.tweet = df_clean.tweet.apply(lambda x: re.sub(r'@[\w]+', '@user', str(x)))

# remove url part 1
df_clean.tweet = df_clean.tweet.apply(lambda x: re.sub(r'https?:\/\/\S+', '', str(x)))
# remove url/website that didn't use http, is only checking for .com websites 
# so words that are seperated by a . are not removed
df_clean.tweet = df_clean.tweet.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', str(x)))

# remove {link}
df_clean.tweet = df_clean.tweet.apply(lambda x: re.sub(r'{link}', '', x))
# remove hastags
df_clean.tweet = df_clean.tweet.apply(lambda x: re.sub(r'#[\w]+', '', x))

import emoji 
import demoji

# Define a function to replace emojis with words
def replace_with_words(text):
    return emoji.demojize(text)

# Apply the function to the 'text' column in the DataFrame
df_clean['tweet'] = df['tweet'].apply(replace_with_words)

# Optionally, you can save the modified DataFrame back to a new CSV file
df_clean.to_csv('sarc_data_no_emojis.csv', index=False)  # Replace 'modified_data.csv' with the desired output file path


tknzr = TweetTokenizer()
df_clean['tweet']= df_clean['tweet'].apply(tknzr.tokenize)

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(word_list):
    """remove stopwords from a list of tokens"""
    return [w for w in word_list if w not in STOPWORDS]
    
df_clean['tweet'] = df_clean['tweet'].apply(remove_stopwords)

#WordNet Lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

def lemmatizing(word_list):
    """remove stopwords from a list of tokens"""
    return [lemmatizer.lemmatize(w) for w in word_list]
    
#isuue here
df_clean['tweet'] = df_clean['tweet'].apply(lemmatizing)

#issues
"""
notes:
1. ':' token from emoji being tokenized
2. first example has keyboard emojis that are being tokenized
3. emoji conversion and tokenization
4. non emoji 'expressions' still remain, e.g. :/
"""