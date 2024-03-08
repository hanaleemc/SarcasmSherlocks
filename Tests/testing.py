
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from transformers import DistilBertTokenizer, TFDistilBertModel

#Do not forget to make a separate file for Data_processingTypeII
from preprocessing import text_low_processing


#This code is for iterating tweet_string one by one

DATA_PATH = './Data/train.En.csv'
df = pd.read_csv(DATA_PATH, usecols=['tweet', 'sarcastic'])
df['ProcessedTweet_IV'] = df['tweet'].apply(text_low_processing)
df_IV = df[['ProcessedTweet_IV', 'sarcastic']]
tweet_texts = df_IV['ProcessedTweet_IV'].tolist()
tweet_labels = df_IV['sarcastic'].tolist()

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
embeddings_tweet_1_np = embeddings_tweet_1.numpy()
#print(embeddings_tweet_1_np)
#print(embeddings_tweet_1_np.shape)

# Access label for the first tweet 
tweet_labels_1 = tweet_labels[1]


# The embeddings are stored as TensorFlow tensors. To work with them further,
# we can convert them to numpy arrays
#embeddings_tweet_1_np = embeddings_tweet_1.numpy()
#print(embeddings_tweet_1)

#LSTM Model

lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(1,32),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', Utils.f1_m])
lstm.compile(loss='SigmoidFocalCrossEntropy',optimizer='adam',metrics=['accuracy'])

print(lstm.summary())

lstm.fit(embeddings_tweet_1_np, epochs=10, validation_data=tweet_labels_1) 
#, class_weight={1:4, 0:1}
