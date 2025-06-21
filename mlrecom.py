import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Enhanced page config with theme settings
st.set_page_config(
    page_title="CineMatch AI",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎬"
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# 2. Premium CSS with glass-morphism effects and animations
st.markdown("""

""", unsafe_allow_html=True)

# 3. Load and preprocess data (cached)
@st.cache_data
def load_data():
    df_local = pd.read_csv("shortened_movie_dataset6.csv")
    # Parse release_date flexibly, once
    if 'release_date' in df_local.columns:
        raw = df_local['release_date'].astype(str).str.strip()
        # Try ISO first
        parsed = pd.to_datetime(raw, format='%Y-%m-%d', errors='coerce')
        # Then try DD-MM-YYYY
        parsed = parsed.fillna(pd.to_datetime(raw, format='%d-%m-%Y', errors='coerce'))
        # Fallback to inference
        parsed = parsed.fillna(pd.to_datetime(raw, errors='coerce', infer_datetime_format=True))
        df_local['release_date'] = parsed
    # Create feature_text if missing
    if 'feature_text' not in df_local.columns:
        def create_feature_text(row):
            kw = row.get('keywords', '') if 'keywords' in row.index else ''
            genres = row.get('genres', '') if 'genres' in row.index else ''
            return f"{genres} {kw}"
        df_local['feature_text'] = df_local.apply(create_feature_text, axis=1)
    return df_local

df = load_data()

# 4. Fit TF-IDF vectorizer once (cached)
@st.cache_data
def fit_vectorizer(corpus):
    vect = TfidfVectorizer(stop_words='english')
    vecs = vect.fit_transform(corpus)
    return vect, vecs

vectorizer, movie_vectors = fit_vectorizer(df['feature_text'].fillna(''))

# 5. Premium header with floating animation
st.markdown("""
<div class="header-container">
    <h1 class="floating">🎬 CineMatch AI</h1>
    <p>Discover your perfect movie match with our AI-powered recommendation engine</p>
</div>
""", unsafe_allow_html=True)

# 6. Define era mapping
ERA_OPTIONS = {
    "All Time": None,
    "2000-2010": (2000, 2010),
    "2010-2020": (2010, 2020),
    "2020-2024": (2020, 2024),
    "Classic (Pre-2000)": (None, 1999),
}

# 7. Filters in a premium glass-morphism card
# 7. Optimized form layout for desktop
with st.container():
    with st.expander("", expanded=True):
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="icon">🎯</div>
                <h2>Your Movie Preferences</h2>
            </div>
            
        """, unsafe_allow_html=True)
        
        # Create a 3-column grid for desktop
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.markdown('<div class="form-section"><h3>🎭 Core Preferences</h3></div>', unsafe_allow_html=True)
            selected_mood = st.selectbox(
                "Mood",
                ["Happy", "Sad", "Thrilling", "Mind-Bending", "Romantic", "Nostalgic", "Inspired"],
                index=0
            )
            
            if 'genres' in df.columns:
                unique_genres = sorted({g.strip() for sub in df['genres'].dropna().str.split(',') for g in sub})
            else:
                unique_genres = []
            if unique_genres:
                default_idx = unique_genres.index('Drama') if 'Drama' in unique_genres else 0
                selected_genre = st.selectbox("Genre", unique_genres, index=default_idx)
            else:
                selected_genre = ""
                
            selected_pace = st.selectbox(
                "Pacing",
                ["Slow and Deep", "Medium and Balanced", "Fast and Action-packed"],
                index=1
            )
        
        with col2:
            st.markdown('<div class="form-section"><h3>📺 Viewing Context</h3></div>', unsafe_allow_html=True)
            if 'original_language' in df.columns:
                langs = sorted(df['original_language'].dropna().unique())
                selected_language = st.selectbox("Language", ['All'] + langs, index=0)
            else:
                selected_language = 'All'
                
            watching_with = st.selectbox("Watching With", ["Alone", "Family", "Partner", "Friends"], index=0)
            
            selected_ending = st.selectbox(
                "Ending Type",
                ["Happy", "Realistic / Neutral", "Twist / Cliffhanger", "Sad", "Open to anything"],
                index=4
            )
        
        with col3:
            st.markdown('<div class="form-section"><h3>⏳ Time Period</h3></div>', unsafe_allow_html=True)
            selected_era_label = st.selectbox("Era", list(ERA_OPTIONS.keys()), index=0)
            era_range = ERA_OPTIONS[selected_era_label]
            
            # Add some visual elements to fill space
            st.markdown("""
            <div class="pro-tip">
                <p><b>Pro Tip:</b> For best results, be specific with your preferences. 
                The more details you provide, the better our AI can match you with 
                your perfect movie!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a decorative element
            st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <div style="display: inline-block; background: rgba(255, 75, 75, 0.1); 
                            padding: 15px 20px; border-radius: 50%; font-size: 1.5rem;">
                    🎬
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# 8. Session state initialization for recommendation logic
if 'last_user_input' not in st.session_state:
    st.session_state['last_user_input'] = None
if 'top_pool' not in st.session_state:
    st.session_state['top_pool'] = []
if 'random_sample' not in st.session_state:
    st.session_state['random_sample'] = []
if 'page' not in st.session_state:
    st.session_state['page'] = 0
if 'mode' not in st.session_state:
    st.session_state['mode'] = None

# 9. Build a combined user_input string including filters so recompute triggers on any change
user_input = (
    f"{selected_genre} {selected_mood} {selected_pace} {selected_ending} {watching_with}"
    + f" LANG={selected_language} ERA={selected_era_label}"
)

# 10. Build filter mask function
def build_filter_mask(df, language, era_range):
    mask = np.ones(len(df), dtype=bool)
    # Language filter
    if language and language != 'All' and 'original_language' in df.columns:
        mask &= (df['original_language'] == language)
    # Era filter
    if era_range is not None and 'release_date' in df.columns:
        start, end = era_range
        years = df['release_date'].dt.year
        if start is None and end is not None:
            mask &= (years <= end)
        elif start is not None and end is None:
            mask &= (years >= start)
        else:  # both not None
            mask &= years.between(start, end)
    return mask

# 11. Premium "Find My Perfect Movie" button
if st.button("✨ Find My Perfect Movie", use_container_width=True):
    with st.spinner("🔍 Finding your perfect movie match..."):
        recompute = (st.session_state['last_user_input'] != user_input) or (not st.session_state['top_pool'])
        if recompute:
            # 11.1 Filter DataFrame first
            mask = build_filter_mask(df, selected_language, era_range)
            if mask.sum() == 0:
                st.warning("No movies match your language/time-period filters. Please broaden your filters.")
                st.session_state['top_pool'] = []
                st.session_state['random_sample'] = []
                st.session_state['mode'] = None
                st.session_state['last_user_input'] = user_input
            else:
                df_filtered = df[mask].reset_index()  # original index in column 'index'
                # 11.2 Compute similarity and select top candidates
                user_vector = vectorizer.transform([user_input])
                similarity_full = cosine_similarity(user_vector, movie_vectors).flatten()
                orig_indices = df_filtered['index'].tolist()
                scores_filtered = similarity_full[orig_indices]
                sorted_idx = np.argsort(scores_filtered)[::-1]
                N = min(100, len(sorted_idx))
                top_indices = [orig_indices[i] for i in sorted_idx[:N]]
                st.session_state['top_pool'] = top_indices
                # Reset page/random sample
                st.session_state['page'] = 0
                sample_size = 10
                if len(top_indices) <= sample_size:
                    st.session_state['random_sample'] = top_indices.copy()
                else:
                    st.session_state['random_sample'] = random.sample(top_indices, sample_size)
                st.session_state['mode'] = 'random'
                st.session_state['last_user_input'] = user_input
                st.success(f"Found {mask.sum()} movies after filtering; using top {N} for recommendations.")
        else:
            # Preferences+filters unchanged: reuse existing pool
            st.success("Using previously computed recommendations. Use Refresh / Next / Previous to browse.")

# 12. Display recommendations if available
if st.session_state['top_pool']:
    st.markdown("---")
    st.markdown('<div class="card-header"><div class="icon">🎬</div><h2>Your Recommendations</h2></div>', unsafe_allow_html=True)
    
    # Buttons: Refresh, Previous Page, Next Page
    colr, colp1, colp2 = st.columns([1,1,1])
    with colr:
        if st.button("🔄 Refresh Recommendations", use_container_width=True):
            top_pool = st.session_state['top_pool']
            sample_size = 10
            if len(top_pool) <= sample_size:
                st.session_state['random_sample'] = top_pool.copy()
            else:
                st.session_state['random_sample'] = random.sample(top_pool, sample_size)
            st.session_state['mode'] = 'random'
    with colp1:
        if st.button("⬅️ Previous Page", use_container_width=True):
            st.session_state['mode'] = 'pagination'
            total = len(st.session_state['top_pool'])
            pages = max(1, (total + 9) // 10)
            st.session_state['page'] = (st.session_state['page'] - 1) % pages
    with colp2:
        if st.button("➡️ Next Page", use_container_width=True):
            st.session_state['mode'] = 'pagination'
            total = len(st.session_state['top_pool'])
            pages = max(1, (total + 9) // 10)
            st.session_state['page'] = (st.session_state['page'] + 1) % pages

    # Determine which indices to show
    if st.session_state['mode'] == 'random':
        to_show_indices = st.session_state['random_sample']
    else:
        page = st.session_state['page']
        pool = st.session_state['top_pool']
        start = page * 10
        end = start + 10
        to_show_indices = pool[start:end]
        total = len(pool)
        pages = max(1, (total + 9) // 10)
        st.markdown(f"Page {page+1} of {pages}")

    if not to_show_indices:
        st.warning("No movies to display for these settings.")
    else:
        for idx in to_show_indices:
            if idx < 0 or idx >= len(df):
                continue
            row = df.iloc[idx]
            with st.container():
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    path = row.get('poster_path') or row.get('poster_path')
                    if isinstance(path, str) and path.strip():
                        st.image(f"https://image.tmdb.org/t/p/w500{path}", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                with col2:
                    title = row.get("title") or row.get("name") or "Unknown Title"
                    st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
                    rating = row.get("vote_average", None)
                    if pd.notna(rating):
                        try:
                            num_stars = int(round(float(rating) / 2))
                            stars = "⭐" * num_stars
                        except:
                            stars = ""
                        st.markdown(f'**Rating:** {stars} ({rating})')
                    overview = row.get('overview', '')
                    if isinstance(overview, str) and overview.strip():
                        st.caption(overview)
                    if 'keywords' in row.index and pd.notna(row.get('keywords')):
                        st.markdown("**Keywords:**")
                        keywords = [kw.strip() for kw in str(row['keywords']).split(",") if kw.strip()]
                        if keywords:
                            st.write(", ".join(keywords[:5]))
                    if 'genres' in row.index and pd.notna(row.get('genres')):
                        genres = str(row["genres"]).split(",")
                        for genre in genres[:3]:
                            st.markdown(f'<span class="tag">{genre.strip().title()}</span>', unsafe_allow_html=True)
                    if 'release_date' in row.index and pd.notnull(row.get('release_date')):
                        year = row['release_date'].year
                        st.markdown(f'<span class="tag">📅 {year}</span>', unsafe_allow_html=True)
                    if 'homepage' in row.index and pd.notna(row.get('homepage')):
                        homepage = str(row['homepage'])
                        if homepage.startswith("http"):
                            st.markdown(f'[🔗 Official Website]({homepage})')                           
                    # Watch Trailer button logic
                    if 'trailer_url' in row.index and pd.notna(row.get('trailer_url')):
                        trailer_url = row['trailer_url']
                    else:
                        # fallback to YouTube search
                        title_search = title.replace(" ", "+")
                        trailer_url = f"https://www.youtube.com/results?search_query={title_search}+trailer"

                    # Display trailer button as a clickable link (opens in new tab)
                    st.markdown(f"""
                            <a href="{trailer_url}" target="_blank" rel="noopener noreferrer">
                                <button style="
                                    background: linear-gradient(to right, var(--primary), var(--primary-dark));
                                    color: white;
                                    padding: 10px 20px;
                                    border: none;
                                    border-radius: 8px;
                                    font-weight: 600;
                                    cursor: pointer;
                                    transition: all 0.3s ease;
                                    margin-top: 15px;
                                    display: inline-block;">
                                    ▶️ Watch Trailer
                                </button>
                            </a>
                            """,
                            unsafe_allow_html=True
                        )
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    CineMatch AI • Developed by Sujal Dhandre • Movie data from TMDB • Powered by Streamlit
</div>
""", unsafe_allow_html=True)
