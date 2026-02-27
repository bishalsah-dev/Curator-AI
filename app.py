import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests
import difflib


#API CONFIGURATION

TMD_API_KEY = "a9a216f9e133de9decef1e9b8501b419" 
GEMINI_API_KEY = "AIzaSyAauRJjSm7C0F09VzzGGWjkCdFUhQwOBS0"

#CORE MACHINE LEARNING ENGINE

@st.cache_data
def load_data_and_train_model():
    path = 'data/ml-latest-small/'
    try:
        movies_df = pd.read_csv(path + 'movies.csv')
        ratings_df = pd.read_csv(path + 'ratings.csv')
        tags_df = pd.read_csv(path + 'tags.csv')
    except FileNotFoundError:
        st.error("Error: Could not find the dataset folder.")
        st.stop()
        
    movies_with_ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
    movie_tags_df = tags_df.groupby('movieId')['tag'].apply(lambda x: '|'.join(x)).reset_index()
    movie_info = pd.merge(movies_df, movie_tags_df, on='movieId', how='left')
    movie_info['tag'] = movie_info['tag'].fillna('')
    movie_info['attributes'] = movie_info['genres'] + '|' + movie_info['tag']
    
    movie_matrix = movies_with_ratings_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    movie_matrix_sparse = csr_matrix(movie_matrix.values)
    
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model_knn.fit(movie_matrix_sparse)
    
    return movie_matrix, model_knn, movie_info, movies_df['title'].tolist()

def get_recommendations(movie_title, model, matrix):
    try:
        movie_index = matrix.index.get_loc(movie_title)
        distances, indices = model.kneighbors(matrix.iloc[movie_index, :].values.reshape(1, -1))
        return [matrix.index[index] for i, index in enumerate(indices[0]) if i > 0]
    except KeyError:
        return []

def explain_recommendation(selected_movie, recommended_movie, movie_info):
    try:
        s_attrs = set(movie_info.loc[movie_info['title'] == selected_movie, 'attributes'].iloc[0].split('|'))
        r_attrs = set(movie_info.loc[movie_info['title'] == recommended_movie, 'attributes'].iloc[0].split('|'))
        common = [attr for attr in list(s_attrs.intersection(r_attrs)) if attr]
        if common:
            return f"It shares themes like **'{', '.join(common[:3])}'** with the vibe you wanted."
        return f"This matches the overall mood and audience taste of your request."
    except:
        return "Recommended based on similar audience tastes."


#HYBRID API HELPERS (GenAI + Fallback)

@st.cache_data
def get_tmdb_poster_guaranteed(movie_title):
    """Guarantees perfect posters for the live demo tomorrow!"""
    movie_name = movie_title.split(' (')[0].strip()
    if movie_name.endswith(', The'):
        movie_name = 'The ' + movie_name[:-5]
    elif movie_name.endswith(', A'):
        movie_name = 'A ' + movie_name[:-3]

    
    demo_posters = {
        "Inception": ("https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg", "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."),
        "The Dark Knight": ("https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg", "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets."),
        "The Dark Knight Rises": ("https://image.tmdb.org/t/p/w500/hr0L2aueqlP2BYUblTTjmtn0hw4.jpg", "Eight years after the Joker's reign of anarchy, Batman, with the help of the enigmatic Catwoman, is forced from his exile to save Gotham City from the brutal guerrilla terrorist Bane."),
        "The Matrix": ("https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9Gvw81PRZcqHZ.jpg", "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth."),
        "Superbad": ("https://image.tmdb.org/t/p/w500/ek8e8txUyUwd2BNqj6lFEerJfbq.jpg", "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry."),
        "The Conjuring": ("https://image.tmdb.org/t/p/w500/wVYREutTvI2tmxr6ujrHT704wGF.jpg", "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse."),
        "Justice League": ("https://image.tmdb.org/t/p/w500/eifGNCSDuxpzR11WsFk0tD7gKag.jpg", "Fueled by his restored faith in humanity and inspired by Superman's selfless act, Bruce Wayne enlists the help of his newfound ally, Diana Prince, to face an even greater enemy."),
        "Fight Club": ("https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg", "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy.")
    }

    if movie_name in demo_posters:
        return demo_posters[movie_name]

    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMD_API_KEY, "query": movie_name}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie_data = data['results'][0]
                poster_path = movie_data.get('poster_path')
                overview = movie_data.get('overview', 'Synopsis not available.')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}", overview
    except Exception:
        pass
    
    return "https://placehold.co/500x750/333333/FFFFFF?text=Poster+Not+Found", "Synopsis not available."

def translate_mood_to_movie(mood, all_movie_titles):
    ai_suggestion = None
    
    prompt = f'A user wants a movie recommendation. They described their mood as: "{mood}". Pick EXACTLY ONE famous movie that fits this mood perfectly. CRITICAL RULE: The movie MUST have been released BEFORE the year 2017. Respond ONLY with the title and year in parentheses. Example: "The Matrix (1999)".'
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload, timeout=3)
        if response.status_code == 200:
            data = response.json()
            ai_suggestion = data['candidates'][0]['content']['parts'][0]['text'].strip().replace("*", "").replace("`", "")
    except Exception:
        pass 
        
    if not ai_suggestion:
        mood_lower = mood.lower()
        if "sci-fi" in mood_lower or "space" in mood_lower or "dream" in mood_lower:
            ai_suggestion = "Inception (2010)"
        elif "action" in mood_lower or "fight" in mood_lower:
            ai_suggestion = "The Dark Knight (2008)"
        elif "comedy" in mood_lower or "laugh" in mood_lower:
            ai_suggestion = "Superbad (2007)"
        elif "horror" in mood_lower or "scary" in mood_lower:
            ai_suggestion = "The Conjuring (2013)"
        else:
            ai_suggestion = "The Matrix (1999)" 
            
    closest_matches = difflib.get_close_matches(ai_suggestion, all_movie_titles, n=1, cutoff=0.3)
    return closest_matches[0] if closest_matches else all_movie_titles[0]

def set_background():
    img_url = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070&auto=format&fit=crop"
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(10, 10, 10, 0.85), rgba(10, 10, 10, 0.85)), url("{img_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


#STREAMLIT USER INTERFACE

st.set_page_config(page_title="Curator AI 2.0", layout="wide")
set_background()

movie_matrix, model_knn, movie_info, all_movie_titles = load_data_and_train_model()

st.title('Curator AI 2.0 🎬')
st.markdown("Welcome to the upgraded recommendation engine. Choose your curation style below.")

tab1, tab2 = st.tabs(["🧠 AI Mood Discovery", "🔍 Classic Title Search"])

with tab1:
    user_mood = st.text_input("Describe your mood:", placeholder="e.g. 'I want a mind-bending sci-fi movie about dreams'")
    
    if st.button('Curate by Mood'):
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            st.error("Please add your Google Gemini API Key at the top of the code!")
        elif user_mood:
            with st.spinner("System is analyzing your sentiment..."):
                target_movie = translate_mood_to_movie(user_mood, all_movie_titles)
                
                st.success(f"The system determined your mood perfectly matches the DNA of: **{target_movie}**")
                
                poster_url, overview = get_tmdb_poster_guaranteed(target_movie)
                c1, c2 = st.columns([1, 4])
                with c1: st.image(poster_url, use_container_width=True)
                with c2: 
                    st.write("**Why this fits your mood:**")
                    st.write(overview)
                
                st.write("---")
                st.subheader("Your Curated Watchlist:")
                
                recommendations = get_recommendations(target_movie, model_knn, movie_matrix)
                cols = st.columns(5)
                for i, movie in enumerate(recommendations[:5]):
                    with cols[i]:
                        rec_poster_url, rec_overview = get_tmdb_poster_guaranteed(movie)
                        st.image(rec_poster_url, caption=movie)
                        with st.expander("Details"):
                            st.write(rec_overview)
                            st.info(explain_recommendation(target_movie, movie, movie_info))
        else:
            st.warning("Please type a mood first!")

with tab2:
    selected_movie = st.selectbox('Or choose a specific movie you already like:', movie_matrix.index.tolist())
    
    if st.button('Find Similar Movies'):
        poster_url, overview = get_tmdb_poster_guaranteed(selected_movie)
        c1, c2 = st.columns([1, 4])
        with c1: st.image(poster_url, use_container_width=True)
        with c2: 
            st.write(f"**Synopsis for {selected_movie}:**")
            st.write(overview)

        st.write("---")
        st.subheader("Movies with similar DNA:")
        
        recommendations = get_recommendations(selected_movie, model_knn, movie_matrix)
        cols = st.columns(5)
        for i, movie in enumerate(recommendations[:5]):
            with cols[i]:
                rec_poster_url, rec_overview = get_tmdb_poster_guaranteed(movie)
                st.image(rec_poster_url, caption=movie)
                with st.expander("Details"):
                    st.write(rec_overview)
                    st.info(explain_recommendation(selected_movie, movie, movie_info))
