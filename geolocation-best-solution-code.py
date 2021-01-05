from sklearn.utils import shuffle
from sklearn import preprocessing  # import the preprocessing library
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from data_preprocessor import DataPreprocessor

#### LOADING DATA #####

dataset = pd.read_csv('data/training.txt', header=None, index_col=0)
validation_dataset = pd.read_csv('data/validation.txt', header=None, index_col=0)

dataset = shuffle(dataset)

# I extracted features from the tweet text on the last column and I extracted the features from it.

training_text = dataset.iloc[:, -1].tolist()
training_coordinates = dataset.iloc[:, :-1]

test_data = validation_dataset.iloc[:, -1].tolist()
test_coordinates = validation_dataset.iloc[:, :-1]

# Read submission data and keep the IDs in a separated list

to_predict_df = pd.read_csv('data/test.txt', header=None, index_col=0)
to_predict_values = to_predict_df.iloc[:, -1].values

to_predict_ids_list = to_predict_df.index.tolist()
to_predict_ids = [int(x) for x in to_predict_ids_list]

# I stored the values of dataframe in a list to parse it in the create_kaggle_file function

to_predict_data = to_predict_values.tolist()

#### DATA PREPROCESSING ######

# I imported tqdm to have a progress bar in some of the other tries of data preprocessing

from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

# Tokenize data using tf-idf

tokenizer = Tokenizer(split=' ')

tokenizer.fit_on_texts(training_text + test_data + to_predict_data)
X_train = tokenizer.texts_to_matrix(training_text, mode='tfidf')

# Scale data using L2 norm
scaler = preprocessing.Normalizer(norm='l2')
X_train = scaler.fit_transform(X_train)

##### TRAIN MODEL #########

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# I kept in the grid just the chosen values so it will run just 1 time with the 3-Fold Validation
param_grid = [
    {'alpha': [10 ** (-5)], 'kernel': ['rbf']}
]

# model = KernelRidge(alpha=10**(-5), kernel = 'rbf')

# Grid Search will run K Fold Cross Validation as part of it.
model = GridSearchCV(
    KernelRidge(), param_grid, scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=4,
    cv=3,
    return_train_score=True
)

model.fit(X_train, training_coordinates)

#### PREDICT ON VALIDATION DATA #####

# Prepare test data for prediction by doing the same operations I did on the training dataset so the model can understand them

X_test = tokenizer.texts_to_matrix(test_data, mode='tfidf')
X_test = scaler.fit_transform(X_test)
predicted_coordinates = model.predict(X_test)

train_predicted_coordinates = model.predict(X_train)

# Display Mean squared error and Mean absolute error on training and validation data
print(mean_squared_error(test_coordinates, predicted_coordinates))
print(mean_squared_error(training_coordinates, train_predicted_coordinates))
print('-------')
print(mean_absolute_error(test_coordinates, predicted_coordinates))
print(mean_absolute_error(training_coordinates, train_predicted_coordinates))


#### CREATING THE KAGGLE SUBMISSION FILE #####

def create_kaggle_file(model, file_name):
    print('----- CREATING KAGGLE SUBMISSION FORMAT ----')

    # Initialize dataframe with the necessary columns
    results = pd.DataFrame(columns=['id', 'lat', 'long'])

    to_predict = tokenizer.texts_to_matrix(to_predict_data, mode='tfidf')
    to_predict = scaler.fit_transform(to_predict)

    kaggle_predictions = model.predict(to_predict)

    for i in range(len(to_predict)):
        current_predicted_coordinates = {'id': int(to_predict_ids[i]), 'lat': kaggle_predictions[i][0],
                                         'long': kaggle_predictions[i][1]}
        results = results.append(current_predicted_coordinates, ignore_index=True)

    # I forced the conversion to int32 for ID because after creating the dataframe the IDs were created as floats,
    # inheriting the type from the coordinates, with .00 for every ID

    results = results.astype({'id': 'int32'})
    results.to_csv('results/' + file_name + '.csv', encoding='utf-8', index=False)


create_kaggle_file(model, 'KernelRidge--Norma-L2-TfIdf')
