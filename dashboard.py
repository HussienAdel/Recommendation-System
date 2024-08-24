import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px 

movies = pd.read_csv(r"MovieLens Data\movies.csv", index_col='movieId')

movies = movies.sort_values(by = 'title')
movies = movies.drop_duplicates()

ratings = pd.read_csv(r"MovieLens Data\ratings.csv")

user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

user_similarities = cosine_similarity(user_item_matrix.fillna(0))
user_similarities_df = pd.DataFrame(user_similarities, columns=range(1, 669), index=range(1, 669))



def movie_details(movieId):
    title = movies['title'][movieId]
    genres = movies['genres'][movieId]
    
    return title, genres

def get_movieId(title):
    movieId = movies[movies.title == title].index[0]
    return movieId


def predict_all(userId, movieId):
    similar_users = user_similarities_df[userId]
    similar_ratings = user_item_matrix.loc[similar_users.index, movieId]
    similar_ratings = similar_ratings[similar_ratings.notna()]

    similar_users = similar_users[similar_ratings.index]
    
    return (similar_ratings * similar_users).sum() / similar_users.sum()

def predict_KNN(userId, movieId, k):
    similar_k_users_idx = user_similarities_df[userId].argsort()[::-1][0:k]  # getting the k similar users indeces.
    similar_k_ratings = user_item_matrix.loc[similar_k_users_idx+1, movieId] # getting the k similar users ratings on a certain movie.
    similar_k_ratings = similar_k_ratings[similar_k_ratings.notna()]         #removing the null values from them.

    if similar_k_ratings.empty: # if NO users rate on that movie. -> put the average of all user ratings or put 0.
        return user_item_matrix.loc[userId].mean()
    
    similar_users = user_similarities_df[similar_k_users_idx+1].loc[userId]
    similar_users = similar_users[similar_k_ratings.index]
    
    return sum(similar_k_ratings * similar_users)/sum(similar_users)


def similar_k_users(userId, k=5):
    similar_users = user_similarities_df[userId].sort_values(ascending=False)[0:k] # you can also use ...sort_values()[::-1] to sort the elements in descending order.
    return similar_users.to_dict()


def recommender(userId, movieId, type='KNN', k=10, include=True):
    rate = None
    
    if pd.isna(user_item_matrix[movieId][userId]): # checking if this field is nan or not
        
        s = similar_k_users(userId, k=k)
        if (list(s.values())[1] >= 0.6):
            type = type
        else:
            type = 'all'    
        
        if type.lower() == 'knn':
            rate = predict_KNN(userId, movieId, k)
        elif type.lower() == 'all':
            rate = predict_all(userId, movieId) 
        
        if rate >= 3.5:
            return 'good', rate
        else:
            return 'bad', rate          
    elif include: # include the "aready watched" also 
        rate = user_item_matrix[movieId][userId]
        return "already watched!", rate 
    
def recommended_movies(userId, n_movies):
    user_rates = []
    all_movives = len(movies)
    for i in user_item_matrix.columns:
        if pd.isna(recommender(userId, i, include=False)): # if the movie has aleardy watched, the recommender(include=False) returns 'nan' value
            pass
        else :
            user_rates.append({i: recommender(userId, i, include=False)})

    user_rates.sort(key=lambda x: list(x.values())[0][1], reverse=True)# sort() rturns nan, so this is incorrect ->  sorted_user_rates = user_rates.sort(key=lambda x: list(x.values())[0][1], reverse=True)
    return user_rates [: n_movies]
   

def movies_he_liked(userId):
    watched = ratings[ratings.userId == userId]
    liked = watched.sort_values(by='rating', ascending=False)
    liked_dict = liked.set_index('movieId')['rating'].to_dict()
    return  liked






## Dashbaord ##
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px 

gb = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])



app = Dash(__name__)

app.layout = html.Div([
    
    html.H1(children='Recommendation System for Movies!', style={'textAlign': 'center', 'color':'blue', 'font-size':'40px'}),
    html.H1(children='Select the desired User ID', style={'textAlign': 'left', 'color':'black'}),
    dcc.Dropdown(ratings.userId.unique(), value=35, id='dropdown-selection-userId'),
    dcc.Graph(id='user-list'),
    html.H1(children='Select the desired Movie', style={'textAlign': 'left', 'color':'black'}),
    dcc.Dropdown(movies.title.unique(), value='Toy Story (1995)',id='dropdown-selection-movie-title'),
    html.Div([
        dcc.Graph(id='movie-chart-1', style={'width': '40%'}),
        dcc.Graph(id='movie-rating-chart', style={'width': '40%'}),
        dcc.Graph(id='movie-chart-2', style={'width': '20%'})
        
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div([
        html.Div([
            
            dcc.Dropdown(ratings.userId.unique(), value=35, id='dropdown-selection-userId-2'),
            dcc.Dropdown(movies.title.unique(), value='Toy Story (1995)', id='dropdown-selection-movie-title-2'),
            dcc.Graph(id="user-rate", style={'width': '20%'})

        ], style={'flex': 1, 'padding': '10px'}),  # Add flex and padding styles

        html.Div([
            
            dcc.Dropdown(ratings.userId.unique(), value=35, id='dropdown-selection-userId-3') ,
            dcc.Loading(dcc.Graph(id="recommended-movies"), type="cube")
                 
        ], style={'flex': 1, 'padding': '10px'})  # Add flex and padding styles
        
    ], style={'display': 'flex', 'flexDirection': 'row'})  # Use flexbox to align divs side by side

    
])

@app.callback(
    
    Output('user-list', 'figure'),
    Input('dropdown-selection-userId', 'value')
)
def update_user_movie_list(userId):
    df = movies_he_liked(userId)
    
    # Replace movieId with actual movie titles
    df['movieTitle'] = df['movieId'].apply(lambda x: movie_details(x)[0])
    
    # Create a horizontal bar chart
    fig = px.bar(
        df,
        x='rating',  # Ratings on the x-axis
        y='movieTitle',  # Movie titles on the y-axis
        orientation='h',  # Horizontal bars
        color='rating',  # Color bars by rating
        color_continuous_scale=px.colors.sequential.Plasma,  # Color scale
        title=f"Ratings of User {userId} for Each Movie - (the number of movies he watched is {len(df)})"
    )
    
    # Update layout to maximize the figure and ensure all titles are visible
    fig.update_layout(
        height=800,  # Increase the height of the figure
        margin=dict(l=200, r=20, t=60, b=40),  # Adjust margins; l=left, r=right, t=top, b=bottom
        yaxis=dict(tickmode='linear'),  # Ensures all y-axis labels (movie titles) are shown
        plot_bgcolor='white',  # Background of the plot
        paper_bgcolor='white',  # Background of the entire figure
    )
    
    return fig

@app.callback(
    
    Output('movie-chart-1', 'figure'),
    Output('movie-rating-chart', 'figure'),
    Output('movie-chart-2', 'figure'),
    Input('dropdown-selection-movie-title', 'value')
)

def update_movie_figure(title):
    
    movieId = get_movieId(title)
    
    genre = movie_details(movieId)[1] # OR -> genre = movies.loc[movieId, 'genres']
   

    #df = gb.loc[gb.index == movieId]
    df1 = ratings.groupby('movieId')['userId'].agg(['count'])
    df1 = df1[df1.index == movieId]
    watched = df1['count'][movieId]
    missed = 668 - watched
    
    df_1 = pd.DataFrame({'type':['watched', 'missed'], 'count':[watched, missed]})
    
    
    fig1 = px.pie(df_1, values='count', names='type', hole=0.5)
    
    '''
        # Bar chart for count
        fig1 = px.bar(
            df,
            x='count',  # Movie titles on the x-axis
            y='title',  # Count on the y-axis
            labels={'count': 'Count'},
            title=f'Count of Ratings for {title}.',
            color_discrete_sequence=['#FF5733']
        )
        '''
    df2 = gb.merge(movies, on='movieId')
    df2 = df2[df2.index == movieId]   
     
    fig2 = px.bar(
    df2,
    x='title',  # Movie titles on the x-axis
    y='mean',   # Mean rating on the y-axis
    labels={'mean': 'Mean Rating'},
    title=f'Mean Rating - # of watches: {watched}',
    color_discrete_sequence=['#33C1FF'],
    width=380,
    height=500
)
    fig2.update_yaxes(range=[0, 5])
    #movie = movies[movies.title == title]

    df3 = ratings[ratings.movieId == movieId]
    liked = df3[df3.rating >= 3.5].shape[0]
    disliked = df3[df3.rating < 3.5].shape[0]
    
    df_3 = pd.DataFrame({'rating_category': ['Liked', 'Disliked'], 'count': [liked, disliked]})

    
    fig3 = px.pie(df_3, values='count', names='rating_category', hole=0.5)



    return fig1,fig3, fig2 



@app.callback(
    
    Output('user-rate', 'figure'),
    Input('dropdown-selection-userId-2', 'value'),
    Input('dropdown-selection-movie-title-2', 'value')
)
def update_user_rating_graph(userId, title):
    
    
    state, rate = recommender(userId=userId, movieId=get_movieId(title), type='knn')
    df = pd.DataFrame({'title': [title], 'state': [state], 'rate': [rate]})
    

    fig = px.bar(df, x='title', y='rate', color='state', width=380, height=500)
    fig.update_yaxes(range=[0, 5])


    return fig


@app.callback(
    
    Output('recommended-movies', 'figure'),
    Input('dropdown-selection-userId-3', 'value')
)
def update_recommended_movies(userId):
    
    reco_movies = recommended_movies(userId, 10) #as a [{id:(good or bad, rateing)}, ...]
    
    
    movie_ids = [list(movie.keys())[0] for movie in reco_movies]
    ratings = [list(movie.values())[0][1] for movie in reco_movies]
    statuses = [list(movie.values())[0][0] for movie in reco_movies]
    
    titles = [movie_details(movieId)[0] for movieId in movie_ids]
    genres = [movie_details(movieId)[1] for movieId in movie_ids]
    

    # Define colors based on the status
    colors = ['green' if status == 'good' else 'red' for status in statuses]

    # Create the bar chart
    fig = px.bar(
    
        x=titles,
        y=ratings,
        color=ratings,  # Color bars by rating
        color_continuous_scale=px.colors.sequential.Plasma,  # Color scale
        height=600,
        width=1100


     )
    fig.update_layout(
        
        xaxis_title='Movie Title',
        yaxis_title='Rating' 
        
         )
    
    fig.update_yaxes(range=[0, 5])
    
    return fig



    
'''
    # Update layout to maximize the figure and ensure all titles are visible
    fig.update_layout(
        height=800,  # Increase the height of the figure
        margin=dict(l=200, r=20, t=60, b=40),  # Adjust margins; l=left, r=right, t=top, b=bottom
        yaxis=dict(tickmode='linear'),  # Ensures all y-axis labels (movie titles) are shown
        plot_bgcolor='white',  # Background of the plot
        paper_bgcolor='white',  # Background of the entire figure
    )
'''

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)