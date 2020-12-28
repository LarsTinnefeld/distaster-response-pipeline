# Importing required libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function to read two csv files containing messages and categories and store, merge and return a dataframe.
    Categories are split up into separate category columns with 1 or 0 flags as values
    
    Inputs:
    - csv file and path containing messages
    - csv file and path containing categories
  
    Output:
    - Dataframe "df" containing the data of both input files
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    
    # Splitting the category column into individual category columns
    categories = categories['categories'].str.split(';', expand = True)
    cols = categories.iloc[0].str.split('-').str[0]
    categories.columns = cols
    
    # Extracting 1 or 0 in each category column
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1].astype(int)
        
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    return df

def clean_data(df):
    '''
    Function to take in a dataframe and cleans it by removing empty entries and duplicates
    
    Input: Dataframe
    
    Output: Clean dataframe
    '''
    
    # Need to drop 'original' in order not to drop based on empties in that column
    df.drop('original', axis = 1, inplace = True)
    
    # Deleting empty records
    df.dropna(inplace = True)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    engine = create_engine(database_filename, encoding="UTF-8")
    df.to_sql('responses', engine, index=False)

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()