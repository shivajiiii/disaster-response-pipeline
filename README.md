# Disaster Response Pipeline Project

In an event of a natural disaster it is necessary of disaster response teams to act swift and save lives that are in danger.

This project aims at classifying the messages and alerting the specific disaster reponse teams with the help of machine learning pipeline that pre-process the data from figure eight disaster response dataset.

This also has a web-app that can classify the messages.


## Installations
Please install python and import following packages and libraries: import nltk, numpy, sklearn, TransformerMixin, pickle, unittest, string, nltk, nltk.tokenize, RegexpTokenizer,sent_tokenize, nltk.stem, pandas, sqlalchemy, re, sklearn

### Folder Structure

- app  
| - templates   
| |- master.html    
| |- go.html  
|- run.py   
  
- data  
|- disaster_categories.csv  
|- disaster_messages.csv   
|- process_data.py  
|- DisasterResponse.db   
  
- models  
|- train_classifier.py  
  
- Screenshot_1  
- Screenshot_2 
- Screenshot_3    

- README.md  


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Messages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Messages.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots:

![Screenshot_1.PNG](https://github.com/shivajiiii/disaster-response-pipeline/blob/master/Screenshot_1.png)
![Screenshot_2.PNG](https://github.com/shivajiiii/disaster-response-pipeline/blob/master/Screenshot_2.png)
![Screenshot_3.PNG](https://github.com/shivajiiii/disaster-response-pipeline/blob/master/Screenshot_3.png)

## Licensing, Authors, Acknowledgements
The dataset was provided by Figure-8 and thanks to Udacity for this interesting project.
