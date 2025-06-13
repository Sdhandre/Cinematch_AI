import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Clean, minimal CSS for consistent layout
st.markdown(
    """
    <style>
    :root {
        --primary: #ff4b4b;
        --secondary: #1a1a1a;
        --text: #f0f0f0;
    }
    
    body {
        background-color: #0e1117;
        color: var(--text);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #121212 0%, #0a0a0a 100%);
    }
    
    .stButton>button {
        background-color: #ff4b4b;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e63c3c;
        transform: scale(1.03);
    }
    .stButton>button:active, 
    .stButton>button:focus {
        background-color: #e63c3c !important;
        color: white !important;
    }
    
    .movie-card {
        background: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid var(--primary);
    }
    
    .title {
        color: var(--primary);
        font-weight: 700;
    }
    
    .tag {
        display: inline-block;
        background: rgba(255, 75, 75, 0.2);
        padding: 4px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.8em;
    }
    
    .header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .question-section {
        background: rgba(30, 30, 30, 0.5);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    
    /* Centered container for consistent layout */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 15px;
    }
    
    @media (max-width: 768px) {
        .main-container {
            padding: 0 10px;
        }
        .header h1 {
            font-size: 1.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load your preprocessed movie dataset
@st.cache_data
def load_data():
    return pd.read_csv("shortened_movie_dataset2.csv")

df = load_data()

# App Header
st.markdown(
    """
    <div class="header">
        <h1>🎬 CineMatch AI</h1>
        <p>Discover your perfect movie match</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main content container for consistent centering
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# User Preferences Section
st.subheader("🎯 Tell Us What You're Looking For")
st.caption("We'll find the ideal movie for your mood and situation")

# Using expanders for better organization
with st.expander("🎭 Mood & Genre", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        # 1. Mood
        selected_mood = st.selectbox(
            "What mood are you in?😀",
            ["Happy", "Sad", "Thrilling", "Mind-Bending", "Romantic", "Nostalgic", "Inspired"],
            index=0
        )
    with col2:
        # 2. Genre
        unique_genres = sorted(list(set([g.strip() for sublist in df['genres'].dropna().str.split(',') for g in sublist])))
        selected_genre = st.selectbox(
            "What genre do you want?😶‍🌫️",
            unique_genres,
            index=unique_genres.index('Drama') if 'Drama' in unique_genres else 0
        )
    
    # Optional: Language filter
    if 'original_language' in df.columns:
        available_languages = sorted(df['original_language'].dropna().unique())
        selected_language = st.selectbox(
            "Preferred movie language?",
            ['All'] + available_languages
        )

with st.expander("⏳ Time & Context", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        # 3. Era
        selected_era = st.selectbox(
            "Which time period?⏲️",
            ["AllTime", "2000-2010", "2010-2020", "2020-2024", "Classic (Pre-2000)"],
            index=2
        )
        
        # 4. Pace
        selected_pace = st.selectbox(
            "What pacing?💨",
            ["Slow and Deep", "Medium and Balanced", "Fast and Action-packed"],
            index=1
        )
    with col2:
        # 5. Watching with
        watching_with = st.selectbox(
            "Who are you watching with?🧐",
            ["Alone", "Family", "Partner", "Friends"],
            index=0
        )
        
        # 6. Ending type
        selected_ending = st.selectbox(
            "What kind of ending?💀",
            ["Happy", "Realistic / Neutral", "Twist / Cliffhanger", "Sad", "Open to anything"],
            index=4
        )

# Apply filters
if 'original_language' in df.columns and selected_language != 'All':
    df = df[df['original_language'] == selected_language]

if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%m-%Y', errors='coerce')
    if selected_era == '2000-2010':
        df = df[df['release_date'].dt.year.between(2000, 2010)]
    elif selected_era == '2010-2020':
        df = df[df['release_date'].dt.year.between(2010, 2020)]
    elif selected_era == '2020-2024':
        df = df[df['release_date'].dt.year.between(2020, 2024)]
    elif selected_era == 'All_Time':
        df = df[df['release_date'].dt.year.between(2000, 2024)]
    elif selected_era == 'Classic (Pre-2000)':
        df = df[df['release_date'].dt.year < 2000]

# Create a combined feature text for vectorization
def create_feature_text(row):
    return f"{row['genres']} {row['keywords']}"

# Preprocess dataset for matching
if 'feature_text' not in df.columns:
    df['feature_text'] = df.apply(create_feature_text, axis=1)

vectorizer = TfidfVectorizer(stop_words='english')
movie_vectors = vectorizer.fit_transform(df['feature_text'].fillna(''))

# Create user preference string
user_input = f"{selected_genre} {selected_mood} {selected_pace} {selected_ending} {watching_with}"
user_vector = vectorizer.transform([user_input])

# Submit button
st.markdown("---")
if st.button("✨ Find My Perfect Movie", use_container_width=True):
    with st.spinner("🔍 Finding your perfect movie match..."):
        similarity_scores = cosine_similarity(user_vector, movie_vectors).flatten()
        top_indices = similarity_scores.argsort()[-10:][::-1]
        top_movies = df.iloc[top_indices]

        if top_movies.empty:
            st.warning("😕 We couldn't find perfect matches. Try adjusting your preferences.")
        else:
            st.success(f"🎉 We found {len(top_movies)} great matches for you!")
            st.markdown("---")
            
            # Simple, clean movie cards
            for _, row in top_movies.iterrows():
                with st.container():
                    st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        path = row.get('backdrop_path') or row.get('poster_path')
                        if path and isinstance(path, str):
                            st.image(f"https://image.tmdb.org/t/p/w500{path}", use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                    
                    with col2:
                        st.markdown(f'<div class="title">{row["title"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'**Rating:** ⭐ {row["vote_average"]}')
                        st.caption(row['overview'])


                        if pd.notna(row.get('keywords')):
                            st.markdown("**Keywords:**")
                            keywords = [kw.strip() for kw in str(row['keywords']).split(",")]
                            st.write(", ".join(keywords[:5]))  # Show first 5 keywords

                        
                        st.markdown(f'<span class="tag">{str(row["genres"]).title()}</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="tag">📅 {row["release_date"].year if pd.notnull(row["release_date"]) else "Unknown"}</span>', unsafe_allow_html=True)
                        
                        if pd.notnull(row.get('homepage')) and str(row['homepage']).startswith("http"):
                            st.markdown(f'[🔗 Official Website]({row["homepage"]})')
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")

# Close main container
st.markdown('</div>', unsafe_allow_html=True)