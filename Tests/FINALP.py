import argparse
from keras.callbacks import EarlyStopping  # Add this line

from Model import prepare_datasets, create_model
from Evaluation import plot_history

train_path = './Data/train.En.csv'
test_path = './Data/test.En.csv'
#train_path = 'Train_Dataset.csv'
#test_path = 'Test_Dataset.csv'

def main():

    parser = argparse.ArgumentParser(description='Training LSTM or BiLSTM model')

    parser.add_argument('--pptype', dest='pptype',
                        choices=['PreprocI', 'PreprocII', 'PreprocIII', 'PreprocIV_A', 'PreprocIV_B'],
                        help='{PreprocI, PreprocII, PreprocIII, PreprocIV_A, PreprocIV_B}',
                        type=str, default='None')

    parser.add_argument('--model', dest='model_type',
                        choices=['LSTM', 'BiLSTM'],
                        help='{LSTM, BiLSTM}',
                        type=str, default='LSTM')

    parser.add_argument('--loss', dest='loss_function',
                        choices=['binary_crossentropy', 'weighted_binary_cross_entropy'],
                        help='{binary_crossentropy, weighted_binary_cross_entropy}',
                        type=str, default='binary_crossentropy')

    parser.add_argument('--pos_weight', dest='pos_weight',
                        help='Positive weight for weighted binary cross-entropy loss',
                        type=float, default=None)

    args = parser.parse_args()

    # Pass the pptype argument to prepare_datasets
    train_data, _, tokenizer = prepare_datasets(train_path, test_path, args.pptype)  # Only training data is required

    # Create model based on the selected model type and loss function
    if args.loss_function == 'weighted_binary_cross_entropy':
        if args.pos_weight is None:
            raise argparse.ArgumentError(None, "The --pos_weight argument is required when using weighted_binary_cross_entropy loss.")
        model = create_model(args.model_type, tokenizer, loss_function=args.loss_function, pos_weight=args.pos_weight)
    else:
        model = create_model(args.model_type, tokenizer, loss_function=args.loss_function)

    print(model.summary())

    # Include EarlyStopping to stop the training process early if a monitored metric has stopped improving
    # The patience parameter is the number of epochs to wait for an improvement
    # verbose- Log a message when stopping
    # Restore model weights from the epoch with the best value of the monitored metric
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    # Include validation data
    _, test_data, _ = prepare_datasets(train_path, test_path, args.pptype)

    # saving the metrics results
    results_visual = model.fit(train_data, epochs=20, validation_data=test_data, class_weight={1: 4, 0: 1},
                               callbacks=[early_stopping])

    # plotting the metrics results
    plot_history(results_visual)

if __name__ == "__main__":
    main()
