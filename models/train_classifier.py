import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_filepath):
    """
    This loads the clean data from database
        - Extract X, Y data
        
    Arguments:
    database_filepath (str): File path of database
            
    Returns:
    X (pandas dataframe): Contains messages
    Y (pandas dataframe): Contains categories of disaster
    category_names (list): Name of categories
    """
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    
    
    # Extract X and Y
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = list(Y.columns)
    
    return X, Y, category_names

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    """
    Lemmatize text, removes capitalization and special characters
        
    Arguments:
    text (str): Message string
            
    Returns:
    clean_tokens (list): List of tokens
    """
    detected_urls = re.findall(url_regex, text)

    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # remove punctuation characters and covert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]

    tokens = [WordNetLemmatizer().lemmatize(word).strip() for word in tokens]
    
    return tokens

def build_model():
    """
    Build a pipeline for creating a model
        
    Returns:
    model (GridSearchCV): Machine learning model
    """

    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
	

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100]
    }
    
    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model
        
    Args:
    model (Predicter model): Machine learning model (Predicter)
    X_test (pandas series): Test data set of X
    Y_test (pandas dataframe): Test data set of Y
    category_names (list): Name of categories
    """
    
    # Predict test data
    predict_y = model.predict(X_test)
    
    # Evaluate
    for i in range(len(category_names)):
        category = category_names[i]
        print(category)
        print(classification_report(Y_test[category], predict_y[:, i]))




def save_model(model, model_filepath):
    """
    Saves the model created in a pickle file
        
    Argumentss:
    model (Predicter model): Machine learning model (Predicter)
    model_filepath (str): Location where the model will be saved.
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
