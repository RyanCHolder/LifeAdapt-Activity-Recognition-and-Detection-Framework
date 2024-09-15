import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

"""Basic 1 dimensional CNN model to use as a baseline to compare other models
    Get command line input for which dataset to use (--dataset=dataset_name)"""

# Create the parser
parser = argparse.ArgumentParser(description="Process command line arguments.")

# Add the --dataset argument
parser.add_argument('--dataset', type=str, help='Name of the dataset')

# Parse the arguments
args = parser.parse_args()

#dataset path
data_dir = f"Comb_Data/{args.dataset}.npz"

def create_sequences(X, y, window_size):
    sequences = []
    targets = []
    for i in range(0, len(X) - window_size + 1, window_size): # Iterate in steps of 'window_size' to avoid overlap
        sequences.append(X[i:i+window_size]) # Take the sequence of 30 data points
        targets.append(y[i+window_size-1]) # Target value is last point in the sequence
    return np.array(sequences), np.array(targets)


data = np.load(data_dir, allow_pickle=True)
window_size = 30

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('int')
y_val = y_val.astype('int')
y_test = y_test.astype('int')
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, window_size)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)

# Check the new shapes
# print(X_train_seq.shape)  # (number of sequences, 30, 6)
# print(y_train_seq.shape)  # (number of sequences,)

model = Sequential()

model.add(Input(shape=(window_size,6)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))  # 30 timesteps, 6 features
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer (for multi-class classification, using softmax)
model.add(Dense(y_train_seq.shape[1], activation='softmax'))  # Number of classes based on y_train_seq shape

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, validation_data=(X_val_seq, y_val_seq))
loss, accuracy = model.evaluate(X_test_seq, y_test_seq)
print(f'Test accuracy: {accuracy}')
