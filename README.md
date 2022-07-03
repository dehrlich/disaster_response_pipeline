# Disaster Response Pipeline Project

### Installations:
- In addition to libraries installed as part of the Anaconda distribution you will also need an environment with the following:
    - nltk, flask, plotly, joblib, sqlalchemy, sklearn
    - python verson 3.6 or higher

### Project Motivation:

- To build an a web app capable of taking in disaster related text messages and classify the message as one or more of 40 distinct categories.

- Within the app, I build an ETL pipeline to extract raw text from digital communcation, clean and tokenize the text, and load the resulting dataset into a SQLite database.

- Using the cleaned dataset, I build a Multi-Output Classifier using a Random Forest Classifier ensemble method. This is achieved using SKLearn and the Pipeline class, among other tools. The trained model is then leveraged by the web app to classify new messages.

- The model can by retrained on updated data by following th instructions below.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
