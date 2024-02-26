
#load needed packages 
import numpy as np
import pandas as pd
import csv
import nltk
import emoji
import re
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#Step 0 | Load data [Pandas]
DATA_PATH = './Data/train.En.csv'
df = pd.read_csv(DATA_PATH, usecols= ['tweet','sarcastic'])


#Preprocessing Type II | Convert emotion icons, urls & convert to @user token 
 
def preprocess_type_II(tweet_text):
    if pd.isna(tweet_text) or tweet_text == '':
        return tweet_text
    # Convert emotion icons to their string text
    tweet_text = emoji.demojize(tweet_text)
    
    # Convert URLs to "HTTPURL" token
    tweet_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'HTTPURL', tweet_text)

    # Convert mentions to "@USER" token
    tweet_text = re.sub(r'@\w+', '@USER', tweet_text)

    return tweet_text


# Apply the preprocessing function to the 'Tweet' column
df['ProcessedTweet_II'] = df['tweet'].apply(preprocess_type_II)

#Create new dataframe with processed tweets 
df_II = df[['ProcessedTweet_II', 'sarcastic']]
print(df_II)

#type III | remove multiple punction, same character, contractions 

def preprocess_type_III(tweet_text):
    smiley_mapping = {":-)": "smiley", ":)": "smiley", ":-(": "sad", ":(": "sad", ":D": "playful"}
    if pd.notna(tweet_text):
        for smiley_code, value in smiley_mapping.items():
            tweet_text = tweet_text.replace(smiley_code, value)

        # Remove more than two successive occurrences of any punctuation
        tweet_text = re.sub(r'[^\w\s]{2,}', ' ', tweet_text)

        # Remove more than two successive occurrences of the same character
        tweet_text = re.sub(r'(\w)\1{2,}', r'\1\1', tweet_text)

        # Replace contractions with full forms
        contraction_mapping = {"isn't": "is not", "’cause": "because","You'd": "you would", "I’m": "I am", "Couldn't": "Could not"}  # Customize as needed
        for contraction, full_form in contraction_mapping.items():
            tweet_text = tweet_text.replace(contraction, full_form)

        return tweet_text

# Apply the preprocessing function to the 'tweet' column
df['ProcessedTweet_III'] = df['tweet'].apply(preprocess_type_III)

# Create a new dataframe with processed tweets and 'sarcastic' column
df_III = df[['ProcessedTweet_III', 'sarcastic']]

# Display the result
print(df_III)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#type IV | lemmatizes words, removes stopwords

def preprocess_type_IV(tweet_text, language='english'):
    if pd.notna(tweet_text):
        lemmatizer = WordNetLemmatizer()
        tweet_text = ' '.join([lemmatizer.lemmatize(word) for word in tweet_text.split()])

        # Remove stopwords (for English)
        stop_words = set(stopwords.words(language))
        tweet_text = ' '.join([word for word in tweet_text.split() if word.lower() not in stop_words])

        return tweet_text


# Apply the preprocessing function to the 'Tweet' column
df['ProcessedTweet_IV'] = df['tweet'].apply(preprocess_type_IV)

#Create new dataframe with processed tweets 
df_IV = df[['ProcessedTweet_IV', 'sarcastic']]

print(df_IV)


