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
The main goal of the ETL Disaster Response Pipeline is to categorize messages through an algorithm and displaying the result on the screen. This is helpting to streamline the process of prioritizing messages and providing a fast overview of the potential content (disaster category).

## Approach <a name="approach"></a>
For the development of the application the ETL pipeline was first drafted and tested in Jupyter Notebooks, and then transfered to files which were in the end responsible to perform the training process.
The ETL pipeline consists of following process steps:
- **Extract** the data from a data source from a data source (in this case two csv files)
- **Transform** the data through data cleaning, sparating the category classes in individual labels, merging the information to a combined data table and saving that table in a sql database
- **Load** the data from the sql database, separate the result labels (disaster classes) and the input data (text messages), splitting into train- and test sets...
