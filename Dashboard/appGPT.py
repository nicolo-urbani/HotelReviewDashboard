import dash
from dash import dcc, dash_table
from dash import html
from dash.dependencies import Input, Output, State
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from data_clean import clean_text
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np
from sklearn.pipeline import make_pipeline
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import shap




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load the trained model and vectorizer
model = joblib.load('Hotel_Review_RF_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


df = pd.read_csv('GPTreview.csv')
df = df.reset_index()

df.insert(0, 'id', range(0, 0 + len(df)))



average_rating = df['Reviewer_Score'].mean()
review_number = df.shape[0] 
max_rating=df['Reviewer_Score'].max()
min_rating = df['Reviewer_Score'].min()

df['review_preprocessed'] = df['review'].apply(lambda x: clean_text(x))
review_processed = vectorizer.transform(df['review_preprocessed'])
review_processed_array = review_processed.toarray()

def sentimentAnalysis(review_processed_array):
    sentiment = model.predict(review_processed)
    return sentiment

sentiment = sentimentAnalysis(review_processed)
average_sentiment =  (np.count_nonzero(sentiment == 1) / review_number)*100
sentiment_df = pd.DataFrame(sentiment.T, columns=['sentiment'])
df['sentiment'] = sentiment_df['sentiment']
PAGE_SIZE = 6
pr = df.apply(lambda x : True
            if x['Reviewer_Score'] >= 6 else False, axis = 1)
# Count number of True in the series
positive_review = len(pr[pr == True].index)
positive_review= (positive_review / review_number ) * 100


def generate_shap_explainer_img():  
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(review_processed_array, approximate=True, check_additivity=False)

    list_feature = vectorizer.get_feature_names_out()
    class_names = ["Negative", "Positive"]


    shap.summary_plot(shap_values, feature_names= list_feature, class_names= class_names,  show=False)
    plt.savefig('shap.png')

   
    plt.clf() 
    shap.summary_plot(shap_values[0], review_processed_array, plot_type="bar", class_names= class_names, 
                  feature_names = list_feature, max_display=20,
                 show=False, color="Red")
    plt.savefig('shap-negative.png')

    plt.clf() 

    shap.summary_plot(shap_values[1], review_processed_array, plot_type="bar", class_names= class_names, 
                  feature_names = list_feature, max_display=20,
                 show=False, color="Green")
    plt.savefig('shap-positive.png')

    plt.clf() 

    shap.summary_plot(shap_values[0],  review_processed_array, feature_names = list_feature, show=False)
    plt.savefig('shap-positive-dett.png')

    plt.clf() 

    shap.summary_plot(shap_values[1],  review_processed_array, feature_names = list_feature, show=False)
    plt.savefig('shap-negative-dett.png')

generate_shap_explainer_img()

# Create the Dash application
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1('Hotel Review Sentiment Analysis'),
    html.Div([
        html.Div([
            html.H3(f"Reviewer Average: {average_rating:.2f}"),
            html.H3(f"Number of Reviews: {review_number}"),
            html.H3(f"Highest Rating: {max_rating:.2f}"),
            html.H3(f"Lowest Rating: {min_rating:.2f}"),
            html.H3(f"Sentiment Positive Review: {average_sentiment:.2f}%"),
            html.H3(f"Dataset Positive Review: {positive_review:.2f}%"),
        ], className="six columns"),
        html.Div([
            html.H3('Sentiment Distribution'),
            dcc.Graph(id="graph",
                    #figure=px.pie(sentiment_df, names=['Negative', 'Positive'], hole=.3)),
                    figure = px.pie(sentiment_df, names='sentiment', hole=.3),
            )
        ], className="six columns"),
    ], className="row"),
    html.Div([
        html.Div([
            html.H3('Reviewer Nationality'),
            dcc.Graph(id='countplot-graph', 
              figure = px.histogram(df, x='Reviewer_Nationality', nbins=5, text_auto=True).update_xaxes(categoryorder='total descending'))
    
        ], className="six columns"),

        html.Div([
            html.H3('Review Average Score Distribution'),
            dcc.Graph(id='review-score-graph', 
              figure = px.histogram(df, x="Reviewer_Score"))
        ], className="six columns"),
    ], className="row"),
    html.Div(id='datatable-row-ids-container'),
    html.H2('Hotel Review Explanation'),
    html.Div([
        html.Div([
           dash_table.DataTable(
        id='table-review',
        data=df.to_dict('records'),
        columns=[
            {'name': i,'id': i} for i in df.loc[:,["Reviewer_Score", "Reviewer_Nationality", "review", "sentiment"]]
        ],
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in df.to_dict('records')
        ],
        tooltip_delay=0,
        tooltip_duration=None,
        style_table={'overflowY': 'auto','overflowX': 'auto'},
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,
        },
        css=[{
            'selector': '.dash-spreadsheet td div',
            'rule': '''
                line-height: 15px;
                max-height: 30px; min-height: 30px; height: 30px;
                display: block;
                overflow-y: hidden;
            '''
        }],
        style_cell_conditional=[
            {'if': {'column_id': 'Reviewer_Score'},
            'width': '10%'},
            {'if': {'column_id': 'Reviewer_Nationality'},
            'width': '10%'},
            {'if': {'column_id': 'sentiment'},
            'width': '5%'},
        ],
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        row_deletable=True,
        page_action='native',
        page_current= 0,
        page_size= 10,
    )], className="six columns"),
    html.Div([
           dcc.Loading(
            id='explainer-review-dataset',
            type="default")
        ], className="six columns"),
    ], className="row"),
  
    html.H2('Review Sentiment Explanation'),
    html.Div([
        html.Div([
            dcc.Textarea(
            id='review-input',
            value='This hotel is excellent. The room was spacious and luxurious, the view from the balcony was breathtaking.',
            placeholder='Enter your hotel review...',
            style={'width': '100%', 'height': 100}),
            html.Div(id='prediction-output'),
        ], className="six columns"),

        html.Div([
           dcc.Loading(
            id='explainer-obj',
            type="default")
        ], className="six columns"),
    ], className="row"),
    
    
])

# Define the callback function to predict the sentiment score
@app.callback(
    Output('prediction-output', 'children'),
    Output('explainer-obj', 'children'),
    Input('review-input', 'value'),
    )
   
def predict_sentiment(review):
    if len(review) > 10:
        # Preprocess the review text
        review_cleaned = clean_text(review)
        review_processed = vectorizer.transform([review_cleaned])
        

        # Make the sentiment prediction
        sentiment_pred = model.predict(review_processed)[0]

        review_cleaned = clean_text(review)
        
        c = make_pipeline(vectorizer, model)

        class_names = ["Negative", "Positive"]
        explainer = LimeTextExplainer(class_names=class_names)

        exp = explainer.explain_instance(review_cleaned, c.predict_proba, num_features=2567)

        obj = html.Iframe(
                srcDoc=exp.as_html(),
                width='100%',
                height='250px',
                style={'border': '2px #d3d3d3 solid'},
            )
        if (sentiment_pred == 0):
            return 'Too much Negativity!\n{}'.format(sentiment_pred), obj
        elif (sentiment_pred == 1):
            return 'That\'s GREAT!\n{}'.format(sentiment_pred), obj
    else:
        return 'No review yet \n{}'.format(sentiment_pred), obj
        



@app.callback(
    Output('explainer-review-dataset', 'children'),
    Input('table-review', 'active_cell'))
def generate_explainer_html(active_cell):
        if active_cell is None:
            return "Select a Review to Explain"
        else:   
            c = df.columns.get_loc("review")
            review= pd.DataFrame(df).iloc[active_cell['row_id']] ['review']
           
            
            review_cleaned = clean_text(review)
        
            c = make_pipeline(vectorizer, model)

            class_names = ["Negative", "Positive"]
            explainer = LimeTextExplainer(class_names=class_names)

            exp = explainer.explain_instance(review_cleaned, c.predict_proba, num_features=2567)

            obj = html.Iframe(
                    srcDoc=exp.as_html(),
                    width='100%',
                    height='500px',
                    style={'border': '2px #d3d3d3 solid'},
                )
            return obj  

@app.callback(
    Output('datatable-row-ids-container', 'children'),
    Input('table-review', 'derived_virtual_row_ids'),
    Input('table-review', 'selected_row_ids'),
    Input('table-review', 'active_cell'))
def update_graphs(row_ids, selected_row_ids, active_cell):

    
    selected_id_set = set(selected_row_ids or [])

    if row_ids is None:
        dff = df
        # pandas Series works enough like a list for this to be OK
        row_ids = df['id']
    else:
        dff = df.loc[row_ids]

    active_row_id = active_cell['row_id'] if active_cell else None


    

    r = dff.groupby(['Reviewer_Nationality'], as_index=False).mean('Reviewer_Score')
    r.sort_values('Reviewer_Score',ascending=False)
    d1= pd.DataFrame()
    d1['Reviewer_Nationality']=r.sort_values('Reviewer_Score',ascending=False)['Reviewer_Nationality']
    d1['Reviewer_Score']=r.sort_values('Reviewer_Score',ascending=False)['Reviewer_Score']

    r = dff.groupby(['Reviewer_Nationality'], as_index=False).mean('sentiment')
    r.sort_values('sentiment',ascending=False)
    #r['sentiment'] = df['sentiment'].apply(lambda x: x/review_number*100)
    d2= pd.DataFrame()
    d2['Reviewer_Nationality']=r.sort_values('sentiment',ascending=False)['Reviewer_Nationality']
    d2['sentiment']=r.sort_values('sentiment',ascending=False)['sentiment']

    import nltk
    df['tok'] = df.review_preprocessed.apply(nltk.tokenize.word_tokenize)
    words = df.tok.tolist() 
    words = [word for list_ in words for word in list_]
    word_dist = nltk.FreqDist(words)
   # remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords)

    # output the results
    mcw = pd.DataFrame(word_dist.most_common(10), columns=['Word', 'Frequency'])

    return [
        html.Div([
        html.Div([
            html.H3('Top Raters by Country'),
             dcc.Graph(
            id='reviewer_score' + '--row-ids',
            figure={
                'data': [
                    {
                        'x': d1['Reviewer_Nationality'],
                        'y': d1['Reviewer_Score'],
                        'type': 'bar',
                      
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': 'Reviewer Score'}
                    },
                    'height': 500,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        ),
        ], className="six columns"),

        html.Div([
            html.H3('Percentage of Positive Sentiment by Country'),
           dcc.Graph(
            id='reviewer_score' + '--row-ids',
            figure={
                'data': [
                    {
                        'x': d2['Reviewer_Nationality'],
                        'y': d2['sentiment'],
                        'type': 'bar',
                      
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': 'Sentiment'}
                    },
                    'height': 500,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        ),
        ], className="six columns"),
    ], className="row"),
     html.Div([
        html.Div([
            html.H3('Sentiment Distribution'), 
            dcc.Graph(
                id="graph",
                ##figure=px.pie(dff, names=['Negative', 'Positive'], hole=.3)
                figure = px.pie(dff, names='sentiment', hole=.3),
            ),
        ], className="six columns"),
      


    html.Div([
            html.H3('Most Common Words'),
             dcc.Graph(
            id='frequency-word' + '--row-ids',
            figure={
                'data': [
                    {
                        'y': mcw['Word'],
                        'x': mcw['Frequency'],
                        'type': 'bar',
                
                        'orientation':'h',
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': 'Word Frequency'}
                    },
                    'height': 500,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        ),
        ], className="six columns"),   
    ], className="row"),
    ]


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
