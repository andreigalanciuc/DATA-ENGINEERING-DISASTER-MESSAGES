# DATA-ENGINEERING-DISASTER-MESSAGES

## PROJECT HIGHLIGHT
In this project I apply data engineering skills to create an API (application programming interface) that classifies disaster messages from various sources (Twitter, text messages) into 36 categories. This classification problem is a type of supervised machine learning because the model learns to classify the outcome based on learning from the data given to it.
I.e. what is the message related to: water, food, shelter, money etc.?
The reason is that when disaster happens, there are millions of messages sent and tweeted to inform about it. However, disasters are taken care of different organizations. Food provision might be offered by a certain organization, while putting down fires would be taken care of a different organization. Hence,the utility of this application would be to categorize these messages into various types so that it can be understood what type of aid is necessary for a specific disaster.

## PROJECT STRUCTURE
There are three parts of the project:

1) ETL Pipeline
Extract, transform and load the data. This is concerned with processing the data. Namely I loaded, merged and cleaned the messages and categories dataset. I stored into an SQLite database so that the model can use it in the next step to train.

2) ML Pipeline
The machine learning pipeline is concerned with training the model and testing it. The pipeline includes a text processing part because it deals with text sources as mentioned in the beginning. I also used GridSearchCV to tune the model further and save it as a pickle file.

3) Flask Web App
The `run.py` `process_data` and `train_classifier` are basically the ETL pipeline and ML pipeline included in the terminal workspace to make the app work. `run.py` relates to the interface of the app.

## RUN THE APP
- To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
Run the following command in the app's directory to run your web app.
    `python app/run.py` 
    
Go to http://0.0.0.0:3001/
    
