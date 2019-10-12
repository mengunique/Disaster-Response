import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle

import warnings
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    '''Load data from SQLite database file
    
    Args:
        database_filepath (str): Path of SQLite database file.
    
    Returns:
        X (Series): Message data
        Y (DataFrame): Categories (labels)
        category_names (list): List of category names
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    '''Tokenize text (with stop words removal and lemmatization)
    
    Args:
        text (str): Input message text
    
    Returns:
        tokens (list): List of tokenized words 
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


# Try to add some extra features
# References:
# https://www.quora.com/Natural-Language-Processing-What-are-the-possible-features-that-can-be-extracted-from-text
# https://www.kaggle.com/shaz13/feature-engineering-for-nlp-classification
def count_words(data):
    """Compute number of words
    
    Args:
        data (Series or 1D-array): Text data.
    
    Returns:
        NumPy 1D-arrary: Word count values for each text.
    """
    return np.array([len(text.split()) for text in data]).reshape(-1, 1)


def build_model():
    '''Build a machine learning pipeline and apply grid search
    
    Args:
        None
    
    Returns:
        cv (GridSearchCV): The grid search object for the model
    '''
    # During the tests, Naive Bayes obtains similar results with Random Forest.
    # Choose Naive Bayes here as it is much faster to train.
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('wordcount', FunctionTransformer(count_words, validate=False))
        ])),
        ('clf', MultiOutputClassifier(BernoulliNB()))
    ])
    
    # The default scoring for MultiOutputClassifier considers a prediction correct
    # only when all categories of a test sample are predicted correctly. This is not what we want here.
    # In this disaster response case, we want the messages to be sent to as-many-as-possible
    # relevant parties (so they can work on the problems). Hence the recall for each category is more important.
    
    # So creating a new scorer (biased towards recall) for the grid search.
    scorer = make_scorer(fbeta_score, beta=2, average='micro')
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1,1), (1,2)],
        'clf__estimator__alpha': [0.01, 0.1],
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate and print model performance
    
    Args:
        model (estimator): Model to be evaluated
        X_test (DataFrame): Features in test dataset
        Y_test (DataFrame): Labels in test dataset
        category_names (list): List of category names
    
    Returns:
        None
    '''
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print(category_names[i])
        cat_Y_test = Y_test.iloc[:,i].values
        cat_Y_pred = Y_pred[:,i]

        f1 = f1_score(cat_Y_test, cat_Y_pred)
        precision = precision_score(cat_Y_test, cat_Y_pred)
        recall = recall_score(cat_Y_test, cat_Y_pred)
        print(f'    F1 Score: {f1:.4f}    % Precision: {precision:.4f}    % Recall: {recall:.4f}')


def save_model(model, model_filepath):
    '''Save model to a Python 
    
    Args:
        model (estimator): Model object
        model_filepath (str): Path to save the model
    
    Returns:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()