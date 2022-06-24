import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """" Function to load a dataset from a SQLite datavase table.
    Loads dataset into pandas dataframe, then loads the explanatory
    variable into X and target categories into Y, as well as a list
    of the target category names into the variable 'category_names.'

    Args:
        database_filepath (string): filepath to database

    Returns:
        array: X, array of explanatory variable
        array: Y, multi-class array of target classes
        list: category_names, list of target category names
    """

    engine = create_engine('sqlite:///../data/{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'].values

    # just grab the column names of the multi-class target
    category_names = list(df.columns)[4:]
    Y = df[category_names].values

    return X,Y, category_names


def tokenize(text):
    """ Function to tokenize a text input. Takes a text input
    and split into the simple text, removing punctuation,
    normalized each token by putting it in lower case, striping
    any additional white space, and lemmatizing each resulting
    token. 

    Args:
        text (string): text input
    
    Returns:
        list: list of cleaned tokens
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """ Function to build a machine learning model pipeline.
    Uses the sklearn Pipeline class to chain the transformation
    and model training steps. Uses gridsearch to find optimal
    model parameters over provided ranges.

    Args: 
        None

    Returns:
        sklearn Pipeline object
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [20],
        'clf__estimator__min_samples_split': [3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ Function to evaluate the performance of the model. Uses the trained
    model to predict target categories based on X_test matrix. Prints overall
    model accuracy, followed by precision, recall, f1-score and support for
    each target category.

    Args:
        model: sklearn trained model object
        array: X_test, array of explanatory variable
        array: Y_test, multi-class array of target classes
        list: category_names, list of target category names

    Returns:
        None
    """

    print('---------- Model Evaluation ----------\n')
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    print("Model Accuracy = {}\n".format(accuracy))

    for i, col_name in enumerate(category_names):
        print('Individual Category Metrics: {}\n'.format(col_name))
        print(classification_report(Y_test[:][i], Y_pred[:][i]))


def save_model(model, model_filepath):
    """ Function to export model to a pickle file for future use
    by app to predict message categories.

    Args:
        model: sklearn trained model object
        model_filepath (string): filepath in which to store picked model

    Returns:
        None
    """
    
    joblib.dump(model, model_filepath)


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