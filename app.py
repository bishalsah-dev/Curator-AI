# Import necessary libraries
import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt
import urllib3

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Curator AI Pro", page_icon="🎬", layout="wide")

# --- TMDb API Configuration ---
# IMPORTANT: Replace this with your own TMDb API key
TMD_API_KEY = "a9a216f9e133de9decef1e9b8501b419" 

# --- Caching Data & Model ---
@st.cache_data
def load_data_and_train_model():
    path = 'data/ml-latest-small/'
    movies_df = pd.read_csv(path + 'movies.csv')
    ratings_df = pd.read_csv(path + 'ratings.csv')
    tags_df = pd.read_csv(path + 'tags.csv')
    
    # Clean Year
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    
    # Merge
    movies_with_ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
    movie_tags_df = tags_df.groupby('movieId')['tag'].apply(lambda x: '|'.join(x)).reset_index()
    movie_info = pd.merge(movies_df, movie_tags_df, on='movieId', how='left')
    movie_info['tag'] = movie_info['tag'].fillna('')
    movie_info['attributes'] = movie_info['genres'] + '|' + movie_info['tag']
    
    # Matrix
    movie_matrix = movies_with_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    movie_matrix_sparse = csr_matrix(movie_matrix.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model_knn.fit(movie_matrix_sparse)
    
    return movie_matrix, model_knn, movie_info, movies_df

# --- API Session Setup with Retries ---
def get_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# --- API Functions ---
@st.cache_data
def fetch_trending_movies():
    """Fetches Top 20 movies released in 2024-2025"""
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMD_API_KEY}&primary_release_date.gte=2024-01-01&sort_by=popularity.desc"
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session = get_session()
        response = session.get(url, verify=False, timeout=10) # Added timeout and session
        response.raise_for_status()
        data = response.json()
        return [m['title'] for m in data['results']]
    except Exception as e:
        st.sidebar.error(f"API Connection Error: {str(e)}")
        return []

@st.cache_data
def fetch_movie_details(movie_title, is_tmdb_search=False):
    """Fetches Poster, Overview, Trailer, Cast"""
    if is_tmdb_search:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMD_API_KEY}&query={movie_title}"
    else:
        movie_name = movie_title.split(' (')[0]
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMD_API_KEY}&query={movie_name}"
    
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session = get_session()
        response = session.get(search_url, verify=False, timeout=10) # Added timeout and session
        response.raise_for_status()
        data = response.json()
        
        if data['results']:
            movie_data = data['results'][0]
            movie_id = movie_data['id']
            real_title = movie_data['title']
            poster_path = movie_data.get('poster_path')
            overview = movie_data.get('overview')
            poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://placehold.co/500x750/333333/FFFFFF?text=No+Poster"
            
            # Details
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMD_API_KEY}&append_to_response=videos,credits"
            details_response = session.get(details_url, verify=False, timeout=10)
            details_data = details_response.json()
            
            # Trailer
            video_key = None
            if 'videos' in details_data and details_data['videos']['results']:
                for video in details_data['videos']['results']:
                    if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                        video_key = video['key']
                        break
            
            # Cast & Director & Genres
            director = "Unknown"
            cast = []
            genres = [g['name'] for g in details_data.get('genres', [])]
            
            if 'credits' in details_data:
                crew = details_data['credits'].get('crew', [])
                cast_data = details_data['credits'].get('cast', [])
                for member in crew:
                    if member['job'] == 'Director':
                        director = member['name']
                        break
                cast = [member['name'] for member in cast_data[:3]]
                
            return real_title, poster_url, overview, video_key, director, cast, genres
        else:
            return movie_title, "https://placehold.co/500x750/333333/FFFFFF?text=Not+Found", "No details found.", None, "Unknown", [], []
            
    except Exception as e:
        return movie_title, "https://placehold.co/500x750/333333/FFFFFF?text=Error", f"Error: {str(e)}", None, "Unknown", [], []

# --- Recommendation Logic ---
def get_recommendations(movie_title, model, matrix, movies_df, is_new_movie):
    if not is_new_movie:
        # CLASSIC: Use k-NN
        if movie_title in matrix.index:
            try:
                movie_index = matrix.index.get_loc(movie_title)
                distances, indices = model.kneighbors(matrix.iloc[movie_index, :].values.reshape(1, -1))
                return [matrix.index[index] for i, index in enumerate(indices[0]) if i > 0], "Collaborative Filtering (User Patterns)"
            except:
                return [], "Error"
        return [], "Not Found"
    else:
        # NEW MOVIE: Use Content-Based (Genre Matching)
        # 1. Get genres of the new movie
        _, _, _, _, _, _, genres = fetch_movie_details(movie_title, is_tmdb_search=True)
        if not genres: return [], "No Data"
        
        # 2. Find classic movies with same genres
        primary_genre = genres[0]
        # Filter dataframe for this genre
        fallback_recs = movies_df[movies_df['genres'].str.contains(primary_genre, case=False, na=False)]
        
        if fallback_recs.empty: return [], "No Matches"
        
        # 3. Return 5 random top ones
        return fallback_recs['title'].sample(5).tolist(), f"Content-Based (Matching Genre: {primary_genre})"

def explain_recommendation(selected_movie, recommended_movie, movie_info, strategy):
    if "Collaborative" in strategy:
        try:
            sel_attrs = set(movie_info.loc[movie_info['title'] == selected_movie, 'attributes'].iloc[0].split('|'))
            rec_attrs = set(movie_info.loc[movie_info['title'] == recommended_movie, 'attributes'].iloc[0].split('|'))
            common = list(sel_attrs.intersection(rec_attrs))
            common = [c for c in common if c]
            reason = f"Because you liked '{selected_movie}'."
            if common:
                reason += f" They share themes like: **{', '.join(common[:3])}**."
            return reason
        except:
            return "Recommended based on user rating patterns."
    else:
        # Explanation for new movies
        genre = strategy.split(': ')[1].replace(')', '')
        return f"Since '{selected_movie}' is a **{genre}** movie, we selected this highly-rated classic from our vault."

# --- Helper Functions ---
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
        background-attachment: fixed;
    }}
    .stMarkdown, .stText, .stHeader {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- MAIN APP ---
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # Silence SSL warnings

movie_matrix, model_knn, movie_info, movies_df = load_data_and_train_model()

try:
    set_background('assets/background.jpg')
except:
    pass

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🔍 Curator Controls")
    
    # THE NEW TOGGLE SWITCH
    collection_mode = st.radio(
        "Select Movie Collection:",
        ["🏛️ Classics (1900-2018)", "🔥 Trending Now (2024-2025)"]
    )
    
    st.markdown("---")
    
    if collection_mode == "🏛️ Classics (1900-2018)":
        st.info("Using **k-NN AI Model** on historical data.")
        min_year = int(movies_df['year'].min())
        max_year = int(movies_df['year'].max())
        year_range = st.slider("Filter Year:", min_year, max_year, (1990, max_year))
        
        # Filter the list
        filtered_movies = movies_df[
            (movies_df['year'] >= year_range[0]) & 
            (movies_df['year'] <= year_range[1])
        ]
        display_list = [m for m in filtered_movies['title'] if m in movie_matrix.index]
        is_trending_mode = False
        
    else:
        st.info("Fetching **Live Data** from TMDb API.")
        if not TMD_API_KEY or TMD_API_KEY == "YOUR_API_KEY_HERE":
            st.error("⚠️ API Key missing!")
            display_list = []
        else:
            display_list = fetch_trending_movies()
        is_trending_mode = True

# --- MAIN UI ---
tab1, tab2 = st.tabs(["🎥 Movie Recommender", "📊 Data Analytics"])

with tab1:
    st.title('Curator AI 3.0 🎬')
    
    if is_trending_mode:
        st.markdown("### 🔥 Exploring: Modern Hits (2024-25)")
    else:
        st.markdown("### 🏛️ Exploring: The Classics Library")

    selected_movie_raw = st.selectbox(
        '📽️ Select a Movie:', 
        display_list
    )

    if st.button('🚀 Generate Recommendations'):
        if not TMD_API_KEY or TMD_API_KEY == "YOUR_API_KEY_HERE":
            st.error("⚠️ Please enter your API Key.")
        else:
            # Fetch Details
            real_title, poster, overview, video_key, director, cast, _ = fetch_movie_details(
                selected_movie_raw, 
                is_tmdb_search=is_trending_mode # True if trending, False if classic
            )
            
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(poster, use_container_width=True)
            
            with col2:
                st.header(real_title)
                st.markdown(f"**🎬 Director:** {director}")
                st.markdown(f"**🎭 Cast:** {', '.join(cast)}")
                st.write(f"**📝 Synopsis:** {overview}")
                
                if video_key:
                    st.markdown("### 🍿 Official Trailer")
                    st.video(f"https://www.youtube.com/watch?v={video_key}")

            # --- RECOMMENDATIONS ---
            st.markdown("---")
            st.subheader("✨ AI Curated Picks For You")
            
            # Get recommendations
            recs, strategy = get_recommendations(selected_movie_raw, model_knn, movie_matrix, movies_df, is_trending_mode)
            
            if recs:
                st.caption(f"⚡ Logic: {strategy}")
                cols = st.columns(5)
                for i, movie in enumerate(recs[:5]):
                    with cols[i]:
                        # Recs are always classics from our DB
                        t, p, o, v, d, c, _ = fetch_movie_details(movie, is_tmdb_search=False)
                        st.image(p, caption=t)
                        
                        with st.expander("💡 Why this?"):
                            explanation = explain_recommendation(selected_movie_raw, movie, movie_info, strategy)
                            st.info(explanation)
            else:
                st.warning("No recommendations found.")

with tab2:
    st.header("📊 Dataset Analytics")
    # Simple genre chart
    all_genres = movies_df['genres'].str.split('|', expand=True).stack()
    top_genres = all_genres.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_genres.plot(kind='bar', color='#ff4b4b', ax=ax)
    ax.set_title("Top Genres in Classics Library")
    st.pyplot(fig)
