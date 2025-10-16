# Import necessary libraries
import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import base64
import requests # New library to make API requests

# --- TMDb API Configuration ---
# IMPORTANT: Replace this with your own TMDb API key
TMD_API_KEY = "YOUR_API_KEY_HERE"

# --- Caching our data loading and model training ---
@st.cache_data
def load_data_and_train_model():
    # ... (This function is exactly the same as before)
    path = 'data/ml-latest-small/'
    movies_df = pd.read_csv(path + 'movies.csv')
    ratings_df = pd.read_csv(path + 'ratings.csv')
    tags_df = pd.read_csv(path + 'tags.csv')
    movies_with_ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
    movie_tags_df = tags_df.groupby('movieId')['tag'].apply(lambda x: '|'.join(x)).reset_index()
    movie_info = pd.merge(movies_df, movie_tags_df, on='movieId', how='left')
    movie_info['tag'] = movie_info['tag'].fillna('')
    movie_info['attributes'] = movie_info['genres'] + '|' + movie_info['tag']
    movie_matrix = movies_with_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    movie_matrix_sparse = csr_matrix(movie_matrix.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model_knn.fit(movie_matrix_sparse)
    return movie_matrix, model_knn, movie_info

# --- NEW FUNCTION: Fetch movie poster and overview from TMDb ---
@st.cache_data # Cache the results to avoid repeated API calls
def fetch_movie_details(movie_title):
    """
    Fetches movie poster URL and overview from TMDb API.
    """
    # Extract the movie name without the year
    movie_name = movie_title.split(' (')[0]
    
    # Construct the API request URL
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMD_API_KEY}&query={movie_name}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        
        if data['results']:
            # Get the first result
            movie_data = data['results'][0]
            poster_path = movie_data.get('poster_path')
            overview = movie_data.get('overview')
            
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            else:
                poster_url = "https://placehold.co/500x750/333333/FFFFFF?text=No+Poster" # Placeholder
            
            return poster_url, overview
        else:
            return "https://placehold.co/500x750/333333/FFFFFF?text=Not+Found", "No overview available."
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return "https://placehold.co/500x750/333333/FFFFFF?text=Error", "Could not fetch details."

# --- Background Image and Recommendation Functions (mostly unchanged) ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_recommendations(movie_title, model, matrix):
    # ... (This function is the same)
    try:
        movie_index = matrix.index.get_loc(movie_title)
        distances, indices = model.kneighbors(matrix.iloc[movie_index, :].values.reshape(1, -1))
        neighbor_indices = indices[0]
        recommendations = [matrix.index[index] for i, index in enumerate(neighbor_indices) if i > 0]
        return recommendations
    except KeyError:
        return ["Movie not found in the dataset."]

def explain_recommendation(selected_movie, recommended_movie, movie_info):
    # ... (This function is the same)
    try:
        selected_attrs = set(movie_info.loc[movie_info['title'] == selected_movie, 'attributes'].iloc[0].split('|'))
        recommended_attrs = set(movie_info.loc[movie_info['title'] == recommended_movie, 'attributes'].iloc[0].split('|'))
        common_attributes = [attr for attr in list(selected_attrs.intersection(recommended_attrs)) if attr]
        if common_attributes:
            return f"It shares common themes like **'{', '.join(common_attributes[:3])}'** with your selection."
        else:
            return f"It is recommended because users who liked '{selected_movie}' also liked this movie."
    except (KeyError, IndexError):
        return "Could not generate a detailed explanation."

# --- Load data and train model ---
movie_matrix, model_knn, movie_info = load_data_and_train_model()

# --- MODIFIED Streamlit App Interface ---
st.title('Curator AI 🎬')
set_background('assets/background.jpg')
st.header('Select a Movie to Get Personalized Recommendations')

movie_list = movie_matrix.index.tolist()
selected_movie = st.selectbox('Choose a movie you like:', movie_list)

if st.button('Find Similar Movies'):
    if not TMD_API_KEY or TMD_API_KEY == "a9a216f9e133de9decef1e9b8501b419":
        st.error("Please add your TMDb API key to the code to fetch movie details.")
    else:
        # Display details for the selected movie
        poster_url, overview = fetch_movie_details(selected_movie)
        st.subheader(f"Because you selected: {selected_movie}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(poster_url, use_column_width=True)
        with col2:
            st.write("**Synopsis:**")
            st.write(overview)

        st.write("---") # Separator

        # Get and display recommendations
        recommendations = get_recommendations(selected_movie, model_knn, movie_matrix)
        st.subheader("Here are some movies you might also like:")
        
        # Display recommendations in a 5-column grid
        cols = st.columns(5)
        for i, movie in enumerate(recommendations[:5]):
            with cols[i]:
                rec_poster_url, rec_overview = fetch_movie_details(movie)
                st.image(rec_poster_url, caption=movie)
                with st.expander("See details"):
                    st.write("**Synopsis:**")
                    st.write(rec_overview)
                    explanation = explain_recommendation(selected_movie, movie, movie_info)
                    st.info(explanation)
