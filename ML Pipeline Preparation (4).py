#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle


# In[2]:


# load data from database
engine = create_engine('sqlite:///disaster.db')
df = pd.read_sql_table('messages_disaster', con = engine)


# In[3]:


df.shape


# In[4]:


df


# In[5]:


X = df['message'] 
Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)


# ### 2. Write a tokenization function to process your text data

# In[6]:


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in clean_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[7]:


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 45)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[9]:


pipeline.fit(X_train, y_train)


# In[8]:


def performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))


# In[12]:


performance(pipeline, X_test, y_test)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[13]:


parameters = { #'vect__ngram_range': ((1, 1), (1, 2)), 
              #'vect__max_df': (0.5, 1.0), 
              #'vect__max_features': (None, 5000), 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 
cv = GridSearchCV(pipeline, param_grid=parameters)
cv


# In[14]:


cv.fit(X_train, y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[15]:


performance(cv, X_test, y_test)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[9]:


pipeline2 = Pipeline([
    ('vect', CountVectorizer()),
    ('best', TruncatedSVD()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])


# In[10]:


pipeline2.get_params()


# In[11]:


pipeline2.fit(X_train, y_train)


# In[12]:


performance(pipeline2, X_test, y_test)


# In[17]:


parameters2 = { #'vect__ngram_range': ((1, 1), (1, 2)), 
              #'vect__max_df': (0.5, 1.0), 
              #'vect__max_features': (None, 5000), 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2] }
              #'clf__estimator__min_samples_split': [2, 4]# 


# In[18]:


cv2 = GridSearchCV(pipeline2, param_grid=parameters2)
cv2


# In[19]:


cv2.fit(X_train, y_train)


# In[20]:


performance(cv2, X_test, y_test)


# ### 9. Export your model as a pickle file

# In[23]:


with open('model.pkl', 'wb') as f:
    pickle.dump(cv2, f)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[2]:


# import libraries
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import sys


# In[4]:


def load_data(database_filepath):
    table_name = 'messages_disaster'
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    return X, y

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in clean_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = { #'vect__ngram_range': ((1, 1), (1, 2)), 
              #'vect__max_df': (0.5, 1.0), 
              #'vect__max_features': (None, 5000), 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def train(X,y,model):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 45)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))
    return model

def export_model(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(cv, f)
    
def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model

if __name__ == '__main__':
    data_file = 'disaster.db'  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline

