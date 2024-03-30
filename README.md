LSTM AND BILSTM MODEL TRAINING
This project provides a flexible tool for training LSTM and BiLSTM models for sarcasm detection
task. It supports various preprocessing techniques and model architectures.

LSTM and BiLSTM Model Training Features Multiple Preprocessing Options:
 Supports different preprocessing techniques to prepare text data for model training.
 Configurable Model Types: Allows the selection between LSTM and BiLSTM models, including
extended and polarity-aware variants.
 Customizable Training Parameters: offers control over learning rate
 Early Stopping: Implements early stopping to halt training when the validation loss
ceases to decrease, preventing overfitting.

Requirements
 TensorFlow
 Keras
 Pandas
 NLTK
 TextBlob

Usage
Run the script from the command line, specifying the desired options.
python train_model.py [--pptype PREPROCESSING_TYPE] [--model MODEL_TYPE] [--ext_model
EXTENDED_MODEL_TYPE] [--lr LEARNING_RATE] [--polar_model POLAR_MODEL_TYPE]
Command-Line Arguments
--pptype: Select the preprocessing type. Options: PreprocI, PreprocII, PreprocIII, PreprocIV_A,
PreprocIV_B. Default: None.
--model: Choose between LSTM and BiLSTM models. Default: None.
--ext_model: Specify an extended model type. Options include variations with L2 regularization
and dropout (BiLSTM_L2_D, BiLSTM_D, etc.). Default: None.
--lr: Set the learning rate for the optimizer. Default: 0.001.
--polar_model: Choose a polarity-aware model type (BiLSTM_polar, LSTM_polar). Default: None.

Dataset
The script uses a training and testing dataset located at ./Data/train.En.csv
and ./Data/test.En.csv, respectively. The datasets contain tweets and their corresponding
sentiment labels.

Output
The script will train the specified model on the training dataset, utilizing early stopping
based on validation loss to prevent overfitting. After training, it will output the model's
performance metrics, including F1 score, accuracy, and balanced accuracy. It will also generate
plots of the training history.
