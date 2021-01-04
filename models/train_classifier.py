# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Function to load data from an sql database, stores in a dataframe and splits into input- and output features

    Input: Database and path

    Outputs:
    - Dataframe
    - Input feature vector "messages"
    - Classification matrix (the prediction targets)
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)

    # Need to shrink this dataset for the current exercise due to file size limitations (GitHub) and run time.
    # Under other circumstances the full dataset would be used.
    df = df[:2000]

    X = df['message']
    Y = df.iloc[:, 4:]

    return X, Y


def tokenize(text):
    '''Function that takes in a text splits with white spaces and creates list of words'''
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    word_list = nltk.tokenize.word_tokenize(text)
    
    stop = stopwords.words("english")
    word_list = [t for t in word_list if t not in stop]

    lemmed_list = [WordNetLemmatizer().lemmatize(w) for w in word_list]

    return lemmed_list


def build_model():
    '''
    Function to specify a model and prepares the learning procedure

    Imput: None

    Output: Model which was fit to the data
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])
    
    #parameters = {
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20]
}

    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv

def evaluate_model(model, X_test, Y_test):
    '''
    Function to set up a model (pipeline), split data into train and test sets and train the model

    Inputs:
    - model
    - X_test
    - Y_test
    - category names

    Output: None
    '''

    # Make predictions with test data
    pred_test = model.predict(X_test)

    # Classification report
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], pred_test[:, i]))
        i = i + 1
    accuracy = (pred_test == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """Function to save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
