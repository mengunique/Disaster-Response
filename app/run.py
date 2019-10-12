import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Required by model pipeline
def count_words(data):
    """Compute number of words
    
    Args:
        data (Series or 1D-array): Text data.
    
    Returns:
        NumPy 1D-arrary: Word count values for each text.
    """
    return np.array([len(text.split()) for text in data]).reshape(-1, 1)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Find the numbers of class 1 and class 0 in each category
    class_counts = df.iloc[:,4:].apply(pd.value_counts).fillna(0)
    class_1_counts = class_counts.loc[1]
    class_0_counts = class_counts.loc[0]
    category_names = class_counts.columns
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Stacked bar chart to show class counts
        {
            'data': [
                Bar(
                    x=category_names,
                    y=class_1_counts,
                    name = 'Class 1'
                ),
                Bar(
                    x=category_names,
                    y=class_0_counts,
                    name = 'Class 0',
                    marker = dict(
                        color = 'rgb(255, 143, 43)'
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True,
                    'tickangle': -45
                },
                'barmode' : 'stack'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()