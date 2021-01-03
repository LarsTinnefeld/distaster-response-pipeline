# NLP - Distaster Response Pipeline
### Web application dashboard with NLP based machine learning ETL pipeline
Lars Tinnefeld 2020-12-28

![disaster](https://images.unsplash.com/photo-1545276070-ec815f01c6ec?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1500&q=80)
*Image [Chris Gallagher](https://unsplash.com/@chriswebdog) on Unsplash*

---
## Table of content
1. [Used libraries](#installation)
2. [Objectives](#objectives)
3. [Approach](#approach)
4. [Data](#data)
5. [Data preparation](#preparation)
8. [Instructions](#instruction)

## Libraries <a name="installation"></a>
Following libraries are used in the system development:
- sys
- pandas
- sqlalchemy
- nltk
- re
- pickle
- sklearn

## Objectives <a name="objectives"></a>
The main goal of the ETL Disaster Response Pipeline is to categorize messages through an algorithm and displaying the result on the screen. This is helpting to streamline the process of prioritizing messages and providing a fast overview of the potential content (disaster category). The application is based on NLP processing of existing messages. The machine learning process initially analyses texts which are flagged with disaster category labels and learns in this way to predict which text is linked to which disaster class.

## Approach <a name="approach"></a>
For the development of the application, the ETL pipeline was initially drafted and tested in Jupyter Notebooks, and then transfered to files which were in the end responsible to perform the training process.

**1) The ETL pipeline consists of following process steps:**
- Extract the data from a data source from a data source (in this case two csv files)
- Transform the data through data cleaning, sparating the category classes in individual labels, merging the information to a combined data table and saving that table in a sql database
- Load the data to a sql database
This process is executed in `process_data.py` and prepared in the Jupyter notebook `ETL_Pipeline_Preparation.ipynb`.

**2) Machine learning process:**
- Extract data from database
- Separate the result labels (disaster classes) and input data (text messages)
- Split into train- and test sets
- Set up NLP model
- Train model with train data
- Generate predictions with test data
- Evaluate model
- Adjust parameters to increade prediction accuracy
- Store model for the use in the web application

This process is executed in `train_classifier.py` and prepared in the Jupyter notebook `ML_Pipeline_Preparation.ipynb`.

**3) Flask web application:**
- Form field to read in a text which a user enters
- The text is processed in the backend's model
- The disaster class is highlighted in a table which is displayed to the user

Instructions for how to execute the app is at the end of this README.

## Data <a name="data"></a>
The data and the project idea was provided from [Udacity](https://www.udacity.com/) and [Figure Eight](https://appen.com/).

**messages.csv**
26,248 records

![messages](https://github.com/LarsTinnefeld/distaster-response-pipeline/blob/main/Media/messages_csv.PNG?raw=true)


**messages.csv**
26,248 records

![categories](https://github.com/LarsTinnefeld/distaster-response-pipeline/blob/main/Media/categories_csv.PNG?raw=true)

## Instructions <a name="instructions"></a>

To execute the app follow the instructions:

1. Run the following commands in the project's root directory to set up your database and model.

- Run ETL pipeline: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- Run ML pipeline: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the command line in the app's directory: `python run.py`

3. Open new web browser and go to http://0.0.0.0:3001/
