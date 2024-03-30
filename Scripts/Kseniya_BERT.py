import pandas as pd
from transformers import BertTokenizer, TFBertModel
from transformers import DistilBertTokenizer, TFDistilBertModel

#Do not forget to make a separate file for Data_processingTypeII
from PreprocessTwo import text_low_processing

'''
****************************************************************************************************
'''
#This code is for proceesing one string

#Here we do the low prerocessing of tweets and create a variable with the list of tweets
DATA_PATH = './train.En.csv'
df = pd.read_csv(DATA_PATH, usecols=['tweet', 'sarcastic'])
df['ProcessedTweet_IV'] = df['tweet'].apply(text_low_processing)
df_IV = df[['ProcessedTweet_IV', 'sarcastic']]
tweet_texts = df_IV['ProcessedTweet_IV'].tolist()

# 'bert-base-uncased' is a large model, so it takes a considerable amount of time for many tweets,
#for one tweet is ok
# it has 12 layers with 768 hidden units each!
model_name = 'bert-base-uncased'

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the BERT model
model = TFBertModel.from_pretrained(model_name)

#We can input only one tweet
inputs = tokenizer(tweet_texts[1], return_tensors="tf", padding=True, truncation=True)

outputs = model(**inputs)

# Get the hidden states
hidden_states = outputs.last_hidden_state

# Optionally, obtain the embedding for the [CLS] token (useful for sentence-level tasks!!)
cls_embedding = hidden_states[:, 0, :]

'''
****************************************************************************************************
'''
#This code is for iterating tweet_string one by one

DATA_PATH = './train.En.csv'
df = pd.read_csv(DATA_PATH, usecols=['tweet', 'sarcastic'])
df['ProcessedTweet_IV'] = df['tweet'].apply(text_low_processing)
df_IV = df[['ProcessedTweet_IV', 'sarcastic']]
tweet_texts = df_IV['ProcessedTweet_IV'].tolist()

# We can use different smaller BERT models like:
# model_name = 'google/bert_uncased_L-4_H-512_A-8'
#(with 4 transformer layers, each with hidden state vectors of size 512 and 8 attention heads.)
# model_name = 'google/bert_uncased_L-2_H-128_A-2'

#This model has 6 Transformer layers (compared to 12 in the base BERT model).
#Size of hidden states is the same as with basic BERT, 768.
model_name = 'distilbert-base-uncased'

# Initialize the tokenizer and model specifically for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertModel.from_pretrained(model_name)

# Function to tokenize a tweet and get embeddings
def get_tweet_embeddings(tweet):
    #Check if a particular tweet is actually a string
    if not isinstance(tweet, str):
        print("Tweet is not a string:", tweet)
        return None
    # Tokenize the tweet
    inputs = tokenizer(tweet, return_tensors="tf", padding=True, truncation=True, max_length=512)

    # Generate embeddings
    outputs = model(**inputs)

    # Get the hidden states (embeddings for each token)
    hidden_states = outputs.last_hidden_state

    return hidden_states

# Dictionary to store embeddings for each tweet
tweet_embeddings = {}


# Process each tweet; [0:10] first 10 tweets!
for i, tweet in enumerate(tweet_texts[0:10]):
    embeddings = get_tweet_embeddings(tweet)
    tweet_embeddings[f"tweet_{i + 1}"] = embeddings

# Access embeddings for the first tweet
embeddings_tweet_1 = tweet_embeddings["tweet_1"]

# The embeddings are stored as TensorFlow tensors. To work with them further,
# we can convert them to numpy arrays
embeddings_tweet_1_np = embeddings_tweet_1.numpy()


