import argparse
from keras.callbacks import EarlyStopping

from Model import prepare_datasets, create_model
from Extended_model import create_ext_model
from Model_with_polarity import prepare_datasets_polar, create_model_polarity
from Evaluation import plot_history

train_path = './Data/train.En.csv'
test_path = './Data/test.En.csv'

def main():

    parser = argparse.ArgumentParser(description='Training LSTM or BiLSTM model')

    parser.add_argument('--pptype', dest='pptype',
                        choices=['PreprocI','PreprocII', 'PreprocIII', 'PreprocIV_A', 'PreprocIV_B'],
                        help='{PreprocI, PreprocII, PreprocIII, PreprocIV_A, PreprocIV_B}',
                        type=str, default='None')

    parser.add_argument('--model', dest='model_type',
                        choices=['LSTM', 'BiLSTM'],
                        help='{LSTM, BiLSTM}',
                        type=str, default='None')

    parser.add_argument('--ext_model', dest='ext_model_type',
                        choices=['BiLSTM_L2_D', 'BiLSTM_D', 'BiLSTM_simple', 'BiLSTM_simple_D',
                                 'LSTM_L2_D', 'LSTM_D', 'LSTM_simple', 'LSTM_simple_D'],
                        help='{BiLSTM_L2_D, BiLSTM_D, BiLSTM_simple, BiLSTM_simple_D, '
                             'LSTM_L2_D, LSTM_D, LSTM_simple, LSTM_simple_D}',
                        type=str, default='None')

    parser.add_argument('--lr', dest='learning_rate',
                        help='Learning rate for the optimizer',
                        type=float, default=0.001)

    parser.add_argument('--polar_model', dest='polar_model_type',
                        choices=['BiLSTM_polar', 'LSTM_polar'],
                        help='{BiLSTM_polar, LSTM_polar}',
                        type=str, default='None')

    args = parser.parse_args()

    # Determine which model function to call based on the provided arguments
    if args.model_type != 'None':
        #Preparing dataset
        train_data, test_data, tokenizer = prepare_datasets(train_path, test_path, args.pptype)
        # Creating the model
        model = create_model(args.model_type, tokenizer, args.learning_rate)
        #Checking if the correct model type is chosen
        print("model_type:", args.model_type)
    elif args.ext_model_type != 'None':
        train_data, test_data, tokenizer = prepare_datasets(train_path, test_path, args.pptype)
        model = create_ext_model(args.ext_model_type, tokenizer, args.learning_rate)
        print("ext_model_type:", args.ext_model_type)
    elif args.polar_model_type != 'None':
        train_data, test_data, tokenizer = prepare_datasets_polar(train_path, test_path, args.pptype)
        model = create_model_polarity(args.polar_model_type, tokenizer, args.learning_rate)
        print("polar_model_type:", args.polar_model_type)
    else:
        raise ValueError("No model type specified.")

    print(model.summary())

    #Include EarlyStopping to stop the training process early if a monitored metric has stopped improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    #Saving the metrics results
    results_visual = model.fit(train_data, epochs=20, validation_data=test_data, class_weight={1: 4, 0: 1},
                               callbacks=[early_stopping])

    #Plotting the metrics results
    plot_history(results_visual)

    # Getting the best F1 score, accuracy and balanced accuracy
    best_f1_score = max(results_visual.history['val_f1_m'])
    best_accuracy = max(results_visual.history['val_accuracy'])
    best_balanced_accuracy = max(results_visual.history['val_balanced_accuracy'])
    print(f"Best F1 Score: {round(best_f1_score,4)}")
    print(f"Best Accuracy: {round(best_accuracy, 4)}")
    print(f"Best Balanced Accuracy: {round(best_balanced_accuracy,4)}")

if __name__ == "__main__":
    main()
