import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout , Conv1D, MaxPooling1D 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



# -----------------------------------------------------------------------------------------------------------------------------------
#                                                  TASK - 1
# -----------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------
# Dataset 1: LSTM Model for Emoticon Data
# ----------------------------------------

# Load Dataset 1 (Emoticon Data)
# Load datasets
data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\train\\train_emoticon.csv')
val_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\valid\\valid_emoticon.csv')
test_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\test\\test_emoticon.csv')

# Step 1: Tokenize the emoji sequences
tokenizer = Tokenizer(char_level=True)  # Treat each emoji as a character
tokenizer.fit_on_texts(data['input_emoticon'])
tokenizer.fit_on_texts(val_data['input_emoticon'])
tokenizer.fit_on_texts(test_data['input_emoticon'])

X_seq = tokenizer.texts_to_sequences(data['input_emoticon'])
X_seq_test = tokenizer.texts_to_sequences(test_data['input_emoticon'])

# Step 2: Pad the sequences
max_seq_length = max(len(seq) for seq in X_seq)
X_padded = pad_sequences(X_seq, maxlen=max_seq_length, padding='post')
X_padded_test = pad_sequences(X_seq_test, maxlen=max_seq_length, padding='post')

# Step 3: Convert labels to numpy array
y = np.array(data['label'])

# Step 4: Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Step 5: Function to build the LSTM model
def build_model(input_length, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=16, input_length=input_length))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 6: Build and train model on full training data
model = build_model(input_length=max_seq_length, vocab_size=len(tokenizer.word_index))
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val), verbose=0)

# Step 7: Predict on test data
test_predictions = (model.predict(X_padded_test) > 0.5).astype("int32")

# Step 8: Save the predictions to a file
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred[0]}\n")

save_predictions_to_file(test_predictions, "pred_emoticon.txt")

# ----------------------------------------
# Dataset 2: SVM Model for Feature Data
# ----------------------------------------

# Load Dataset 2 (Feature Data)
train_data = np.load('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\train\\train_feature.npz')
valid_data = np.load('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\valid\\valid_feature.npz')
test_data = np.load('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\test\\test_feature.npz')

# Prepare the training and validation features
train_features_flat = train_data['features'].reshape(train_data['features'].shape[0], -1)  # Flatten the features
valid_features_flat = valid_data['features'].reshape(valid_data['features'].shape[0], -1)  # Flatten the features

# Train SVM on full training data for dataset 2
svm_model_full = SVC(kernel='linear', random_state=42)
svm_model_full.fit(train_features_flat, train_data['label'])

# Predict on test data for dataset 2
test_features_flat = test_data['features'].reshape(test_data['features'].shape[0], -1)  # Flatten the features
svm_test_predictions = svm_model_full.predict(test_features_flat)


# Save Dataset 2 predictions to a file
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

# Save Dataset 2 predictions to a file
save_predictions_to_file(svm_test_predictions, "pred_feat.txt")


# ---------------------------------------------
# Dataset 3: LSTM Model for Text Sequence Data
# ---------------------------------------------

# Load train, validation, and test data
train_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\train\\train_text_seq.csv')
valid_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\valid\\valid_text_seq.csv')
test_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\test\\test_text_seq.csv')

# Convert input strings to sequences of integers
X_train = train_data['input_str'].apply(lambda x: [int(c) for c in x]).tolist()
y_train = train_data['label'].tolist()
X_val = valid_data['input_str'].apply(lambda x: [int(c) for c in x]).tolist()
y_val = valid_data['label'].tolist()
X_test = test_data['input_str'].apply(lambda x: [int(c) for c in x]).tolist()

# Define maximum sequence length
max_sequence_length = 50  # Based on dataset description

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_sequence_length, padding='post')
X_val = pad_sequences(X_val, maxlen=max_sequence_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_sequence_length, padding='post')

# Convert to numpy arrays
X_train = np.array(X_train, dtype=np.int32)
X_val = np.array(X_val, dtype=np.int32)
X_test = np.array(X_test, dtype=np.int32)
y_train = np.array(y_train, dtype=np.int32)
y_val = np.array(y_val, dtype=np.int32)

# Define the model (Use the full training set for prediction on test)
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=16, input_length=max_sequence_length))  # 10 possible digits (0-9)
model.add(Conv1D(filters=25, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(35, dropout=0.4, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using all the training data
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Predict test data using the trained model
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

# Save the predictions to a file
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred[0]}\n")

save_predictions_to_file(y_test_pred, "pred_text.txt")



# -------------------------------------------------------------------------------------------------------------------------------------
#                                                     TASK - 2
# -------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------
# Combined Model of 3 Datasets
# ---------------------------------------------

# Load Emoticons as Features Dataset
emoticons_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\train\\train_emoticon.csv')
X_emoticons = emoticons_data.iloc[:, :-1]  # All columns except label
y_emoticons = emoticons_data['label']  # Label column

# Load Validation Emoticons Dataset
emoticons_val_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\valid\\valid_emoticon.csv')
X_emoticons_val = emoticons_val_data.iloc[:, :-1]  # All columns except label
y_emoticons_val = emoticons_val_data['label']

# One-Hot Encoding for Emoticons
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_emoticons_encoded = onehot_encoder.fit_transform(X_emoticons)

# Load Deep Features Dataset (Training and Validation)
deep_features_data = np.load('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\train\\train_feature.npz')
X_deep = deep_features_data['features']  # Shape: (n_samples, 13, 786)

deep_features_val_data = np.load('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\valid\\valid_feature.npz')
X_deep_val = deep_features_val_data['features']  # Shape: (n_samples, 13, 786)

# Flatten the deep features
X_deep_flat = X_deep.reshape(X_deep.shape[0], -1)  # Reshape to (n_samples, 13 * 786)
X_deep_val_flat = X_deep_val.reshape(X_deep_val.shape[0], -1)

# Apply PCA to reduce dimensionality of the deep features
pca = PCA(n_components=100)
X_deep_pca = pca.fit_transform(X_deep_flat)
X_deep_val_pca = pca.transform(X_deep_val_flat)

# Load Text Sequence Dataset (Training and Validation)
text_sequence_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\train\\train_text_seq.csv')
X_text = text_sequence_data['input_str'].apply(lambda x: [int(c) for c in x]).tolist()
y_text = text_sequence_data['label']

text_sequence_val_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\valid\\valid_text_seq.csv')
X_text_val = text_sequence_val_data['input_str'].apply(lambda x: [int(c) for c in x]).tolist()
y_text_val = text_sequence_val_data['label']

# Pad sequences for the text input
max_sequence_length = 50
X_text = pad_sequences(X_text, maxlen=max_sequence_length, padding='post')
X_text_val = pad_sequences(X_text_val, maxlen=max_sequence_length, padding='post')

# Combine features: Emoticons + PCA Deep Features + Text Features (Training)
X_combined = np.hstack((X_emoticons_encoded, X_deep_pca, X_text))

# Combine features: Emoticons + PCA Deep Features + Text Features (Validation)
X_combined_val = np.hstack((onehot_encoder.transform(X_emoticons_val), X_deep_val_pca, X_text_val))

# List to store validation accuracies
validation_accuracies = []
training_accuracies = []

# List of training set sizes (20%, 40%, 60%, 80%, 100%)
training_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

# Regularization strength (smaller values = stronger regularization)
regularization_strength = 10

for size in training_sizes:
    if size < 1.0:
        # Take a subset of the training data based on the current size
        X_train, _, y_train, _ = train_test_split(X_combined, y_emoticons, train_size=size, random_state=42)
    else:
        # For 100% of data, use the entire training set
        X_train, y_train = X_combined, y_emoticons

    # Train Logistic Regression model with L2 regularization
    logistic_model = LogisticRegression(max_iter=1000, C=1/regularization_strength, penalty='l2')
    logistic_model.fit(X_train, y_train)

    # Evaluate the model on training data
    y_train_pred = logistic_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Evaluate the model on validation data
    y_val_pred = logistic_model.predict(X_combined_val)
    val_accuracy = accuracy_score(y_emoticons_val, y_val_pred)

    # Store accuracies
    training_accuracies.append(train_accuracy * 100)  # Store as percentage
    validation_accuracies.append(val_accuracy * 100)  # Store as percentage

# Load the Test Datasets
# Load Emoticons Test Dataset
emoticons_test_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\test\\test_emoticon.csv')
X_emoticons_test = emoticons_test_data  # All columns are features in test data

# Load Deep Features Test Dataset
deep_features_test_data = np.load('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\test\\test_feature.npz')
X_deep_test = deep_features_test_data['features']  # Shape: (n_samples, 13, 786)

# Load Text Sequence Test Dataset
text_sequence_test_data = pd.read_csv('C:\\Users\\mithu\\Downloads\\mini-project-1\\mini-project-1\\datasets\\test\\test_text_seq.csv')
X_text_test = text_sequence_test_data['input_str'].apply(lambda x: [int(c) for c in x]).tolist()

# Preprocess the Test Data
# One-Hot Encode Emoticons Test Data
X_emoticons_test_encoded = onehot_encoder.transform(X_emoticons_test)

# Flatten Deep Features Test Data and Apply PCA
X_deep_test_flat = X_deep_test.reshape(X_deep_test.shape[0], -1)  # Reshape to (n_samples, 13 * 786)
X_deep_test_pca = pca.transform(X_deep_test_flat)  # Apply PCA transformation

# Pad the Text Sequences for Test Data
X_text_test_padded = pad_sequences(X_text_test, maxlen=max_sequence_length, padding='post')

# Combine features: Emoticons + PCA Deep Features + Text Features (Test)
X_combined_test = np.hstack((X_emoticons_test_encoded, X_deep_test_pca, X_text_test_padded))

# Predict Test Labels using the Trained Logistic Regression Model
y_test_pred = logistic_model.predict(X_combined_test)

# Step 8: Save the predictions to a file in the specified format
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")  # Save as a new line for each prediction

save_predictions_to_file(y_test_pred, "pred_combined.txt")
    
    
