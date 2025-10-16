# Import necessary libraries
import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import base64 # New library to work with local images

# --- Caching our data loading and model training ---
@st.cache_data
def load_data_and_train_model():
    # ... (The rest of this function is exactly the same as before)
    # --- Load Data ---
    path = 'data/ml-latest-small/'
    movies_df = pd.read_csv(path + 'movies.csv')
    ratings_df = pd.read_csv(path + 'ratings.csv')
    tags_df = pd.read_csv(path + 'tags.csv')

    # --- Data Cleaning and Merging ---
    movies_with_ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
    movie_tags_df = tags_df.groupby('movieId')['tag'].apply(lambda x: '|'.join(x)).reset_index()
    
    movie_info = pd.merge(movies_df, movie_tags_df, on='movieId', how='left')
    movie_info['tag'] = movie_info['tag'].fillna('')
    movie_info['attributes'] = movie_info['genres'] + '|' + movie_info['tag']
    
    # --- Create the User-Item Matrix ---
    movie_matrix = movies_with_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    movie_matrix_sparse = csr_matrix(movie_matrix.values)

    # --- Train the k-NN Model ---
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model_knn.fit(movie_matrix_sparse)
    
    return movie_matrix, model_knn, movie_info

# --- NEW: Function to read a local image and encode it ---
@st.cache_data # Cache the image so it doesn't reload every time
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- MODIFIED: Function to set the background using the local image ---
def set_background(image_file):
    """
    Sets a background image for the Streamlit app.
    """
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Define Recommendation and Explanation Functions (these are unchanged) ---
def get_recommendations(movie_title, model, matrix):
    # ... (This function is the same)
    try:
        movie_index = matrix.index.get_loc(movie_title)
        distances, indices = model.kneighbors(matrix.iloc[movie_index, :].values.reshape(1, -1))
        neighbor_indices = indices[0]
        
        recommendations = []
        for i, index in enumerate(neighbor_indices):
            if i > 0:
                recommendations.append(matrix.index[index])
        return recommendations
    except KeyError:
        return ["Movie not found in the dataset."]

def explain_recommendation(selected_movie, recommended_movie, movie_info):
    # ... (This function is the same)
    try:
        selected_attrs = set(movie_info.loc[movie_info['title'] == selected_movie, 'attributes'].iloc[0].split('|'))
        recommended_attrs = set(movie_info.loc[movie_info['title'] == recommended_movie, 'attributes'].iloc[0].split('|'))
        common_attributes = list(selected_attrs.intersection(recommended_attrs))
        common_attributes = [attr for attr in common_attributes if attr]

        if common_attributes:
            explanation = f"It shares common themes like **'{', '.join(common_attributes[:3])}'** with your selection."
            return explanation
        else:
            return f"It is recommended because users who liked '{selected_movie}' also liked this movie."
    except (KeyError, IndexError):
        return "Could not generate a detailed explanation."

# --- Load data and train model ---
movie_matrix, model_knn, movie_info = load_data_and_train_model()

# --- Streamlit App Interface ---
st.title('Curator AI 🎬')

# Call the function to set the background using your local file
set_background('assets/background.jpg')

st.header('Select a Movie to Get Personalized Recommendations')

movie_list = movie_matrix.index.tolist()
selected_movie = st.selectbox('Choose a movie you like:', movie_list)

if st.button('Find Similar Movies'):
    st.write(f"Finding recommendations based on your selection: **{selected_movie}**")
    recommendations = get_recommendations(selected_movie, model_knn, movie_matrix)
    st.subheader("Here are some movies you might also like:")
    for movie in recommendations[:5]:
        explanation = explain_recommendation(selected_movie, movie, movie_info)
        st.markdown(f"**- {movie}**")
        st.info(explanation)