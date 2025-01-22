# -Multi-Model-Framework-for-Sequential-and-Text-Data-Analysis

This project contains implementations of machine learning models for binary classification using three datasets (Emoticon, Deep Feature, and Text Sequence datasets). The project trains models individually on each dataset and then combines features from all datasets to train a unified model.

## Structure of the Project

### Datasets
1. **Dataset 1: Emoticon Dataset** 
   - A sequence of emoticons is used as input.
   - Binary labels (0 or 1) are used for classification.
   
2. **Dataset 2: Deep Feature Dataset**
   - Input features are vectors of deep embeddings.
   - These features are flattened for processing.

3. **Dataset 3: Text Sequence Dataset**
   - Input is a string of digits, with each character representing a feature.
   - Binary labels (0 or 1) are used for classification.

### Tasks

#### Task 1: Individual Model Training

1. **LSTM Model for Emoticon Dataset**
   - Tokenizes the emoticons as sequences and pads them for input into an LSTM.
   - Trained using a sequential model with LSTM and Dense layers.
   - Predictions are made on the test set and saved to a file.

2. **SVM Model for Deep Feature Dataset**
   - Trains an SVM model using the flattened deep feature vectors.
   - The model predicts binary labels on the test set.

3. **LSTM Model for Text Sequence Dataset**
   - Converts text sequences (strings of digits) to integer sequences.
   - Sequences are padded to a fixed length and used as input to an LSTM model.
   - Predictions are made and saved to a file.

#### Task 2: Combined Model

- Combines features from all three datasets:
  1. One-hot encoded emoticons.
  2. PCA-reduced deep feature vectors.
  3. Padded sequences from the text sequence dataset.
  
- Trains a Logistic Regression model with L2 regularization on the combined features.
- The model is evaluated on validation data for various training set sizes (20%, 40%, 60%, 80%, and 100%).
- Predictions are made on the test set using the combined model.

## File Descriptions

1. **train_emoticon.csv, valid_emoticon.csv, test_emoticon.csv**: Emoticon dataset files.
2. **train_feature.npz, valid_feature.npz, test_feature.npz**: Deep feature dataset files.
3. **train_text_seq.csv, valid_text_seq.csv, test_text_seq.csv**: Text sequence dataset files.
4. **pred_emoticon.txt**: Predictions for the Emoticon dataset.
5. **pred_feat.txt**: Predictions for the Deep Feature dataset.
6. **pred_text.txt**: Predictions for the Text Sequence dataset.
7. **pred_combined.txt**: Predictions for the combined model.

## How to Run

1. Install dependencies:
    ```bash
    pip install numpy pandas scikit-learn tensorflow
    ```

2. Load the datasets and run the models:
    - Modify file paths to the datasets in the code if necessary.
    - The script will load datasets, train models, and output predictions.

3. Save the predictions:
    - Predictions for each model will be saved in respective `.txt` files.

## Model Descriptions

- **LSTM Model**: Used for sequence data like Emoticon and Text Sequence datasets. 
- **SVM Model**: Used for the Deep Feature dataset due to its high-dimensional feature space.
- **Logistic Regression**: Used for the combined dataset, combining features from all datasets.

## Evaluation

- Training and validation accuracies are computed for different sizes of training data.
- The combined model uses Logistic Regression with L2 regularization to evaluate the performance on the combined feature set.

## Results

- Test set predictions are saved for each dataset model individually and the combined model.
