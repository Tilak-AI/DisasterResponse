# Udacity-Disaster Response Pipeline

This repository contains ETL, NLP pipelines and a Flask app that I have created for my Udacity's Data Science Nanodegree.

## Table of Contents  
[Installation](#installation)  
[Project Motivation](#motivation)  
[File Descriptions](#files)  
[Results](#results)  
[Licensing & Acknowledgements](#licensing)  

<a name="installation"/></a>
## Installation

The below libraries have been installed alongside a standard distribution of Python:

```bash
pandas
numpy
nltk
sqlalchemy
sklearn
sys
re
flask
langdetect
iso639
iso-639
plotly
json
```

The code should run with no issues using Python 3.7.

Please ensure that you have up-to-date package before you want to use the below code!

<a name="motivation"/></a>
## Project Motivation


This code has been written to complete an assignment which forms a part of Udacity's Data Science Nanodegree.

This has been done in 3 below steps:
- to build an ETL pipeline - combine text messages related to disasters with their labels/categories
- to create an NLP pipeline - clean the text, extract features, perform classification, evaluate the model and save it
- to prepare a Flask app - visualise the characteristics of the training data; allow any user to type a message and 
                           run the model to check how it would classified
                           
Please note that the repository DOES NOT contain the PICKLED MODEL, which  needed to make A predictions in the Flask app! 
due to its large size and GitHub's restrictions. 

<a name="files"/></a>
## File Descriptions

Apart from the README file, below is the structure of this repository:

```bash
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # ETL pipeline
|- discover_languages.py # script to detect languages 
|- DisasterResponse.db # database where tables are stored 

- models
|- train_classifier.py # NLP pipeline

- README.md
```

<a name="results"/></a>
## Instruction

To prepare all files needed by the app, run the commands in the following order:

```bash
#  To run ETL pipeline that cleans data and stores in database:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

python data/discover_languages.py

# To run the Model for classification:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
Your data is now ready and your pickled model has been created!

```bash
# Run the following command to run your web app

 python app/run.py
 Launch Internet explore/chrome/safari/opera/firefox Browser and go to  http://0.0.0.0:3001/
```
<a name="licensing"/></a>
## Licensing & Acknowledgements

Please feel free to use my code as per your need!
