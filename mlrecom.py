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

# 2. Premium CSS with glass-morphism effects and animations
st.markdown("""
<style>
    :root {
        --primary: #ff4b4b;
        --primary-dark: #e63c3c;
        --accent: #6a11cb;
        --accent-light: #2575fc;
        --dark: #0a0a0a;
        --darker: #050505;
        --card-bg: rgba(25, 25, 35, 0.7);
        --card-border: rgba(255, 255, 255, 0.1);
    }
    
    body {
        background: linear-gradient(135deg, var(--darker), var(--dark));
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Segoe UI', system-ui, sans-serif;
        color: #f0f0f0;
        overflow-x: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 100%);
        padding-bottom: 3rem;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin: 2rem 0 3rem;
        position: relative;
        z-index: 10;
    }
    
    .header-container h1 {
        font-size: 4rem;
        background: linear-gradient(to right, #ff4b4b, #ff8e53, #6a11cb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(106, 17, 203, 0.3);
    }
    
    .header-container p {
        font-size: 1.4rem;
        margin: 0;
        padding: 0;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 300;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Glass-morphism card styling */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid var(--card-border);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        box-shadow: 0 12px 40px rgba(106, 17, 203, 0.25);
        transform: translateY(-5px);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .card-header h2 {
        font-size: 1.8rem;
        margin: 0;
        background: linear-gradient(to right, #ff8e53, #6a11cb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .card-header .icon {
        font-size: 2rem;
        color: var(--primary);
    }
    
    /* Form elements styling */
    .form-row-group {
        display: flex;
        gap: 20px;
        margin-bottom: 1.5rem;
    }
    
    .form-row {
        flex: 1;
    }
    
stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(30, 30, 40, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
    }
    
    /* Selected option styling */
    .stSelectbox div[data-baseweb="select"] > div > div > div {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Dropdown menu styling */
    div[role="listbox"] div {
        background-color: #1e1e2e !important;
        color: #f0f0f0 !important;
    }
    
    div[role="listbox"] div:hover {
        background-color: #2a2a3a !important;
        color: #ffffff !important;
    }
    
    /* Focus state for better visibility */
    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.3) !important;
    }
    
    /* Selected option in dropdown */
    div[role="option"][aria-selected="true"] {
        background-color: rgba(106, 17, 203, 0.5) !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Form labels */
    .stSelectbox label {
        color: #d4b1ff !important;
        font-weight: 600;
        margin-bottom: 8px;
        display: block;
    }
    
    /* Section headers */
    .form-section h3 {
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 1rem;
        color: #ff8e53;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Pro tip box */
    .pro-tip {
        margin-top: 20px;
        padding: 15px;
        background: rgba(106, 17, 203, 0.15); 
        border-radius: 12px;
        border-left: 4px solid #6a11cb;
    }
    
    .pro-tip p {
        margin: 0;
        font-size: 0.9rem;
        color: #d4b1ff;
    }
    /* Movie card styling */
    .movie-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid var(--card-border);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(106, 17, 203, 0.25);
        border-color: rgba(106, 17, 203, 0.3);
    }
    
    .movie-title {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        color: white;
        font-weight: 700;
        background: linear-gradient(to right, #ff8e53, #6a11cb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .tag {
        display: inline-block;
        background: rgba(106, 17, 203, 0.2);
        border: 1px solid rgba(106, 17, 203, 0.3);
        color: #d4b1ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header-container h1 {
            font-size: 2.8rem;
        }
        
        .header-container p {
            font-size: 1.1rem;
        }
        
        .form-row-group {
            flex-direction: column;
            gap: 10px;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
    }
    
    /* Floating elements */
    .floating {
        animation: floating 6s ease-in-out infinite;
    }
    
    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 10, 15, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, var(--primary), var(--accent));
        border-radius: 4px;
    }
    
    /* Glowing effect for important elements */
    .glow {
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.5);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
        margin-top: 3rem;
    }
        /* Base button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff4b4b, #e63c3c) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
        width: 100%;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    /* Hover effect - Enhanced */
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.4) !important;
        background: linear-gradient(135deg, #ff5e5e, #ff2d2d) !important;
    }
    
    /* Ripple effect on click */
    .stButton > button:active:after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: rgba(255, 255, 255, 0.5);
        opacity: 0;
        border-radius: 100%;
        transform: scale(1, 1) translate(-50%);
        transform-origin: 50% 50%;
        animation: ripple 0.6s ease-out;
    }
    
    @keyframes ripple {
        0% {
            transform: scale(0, 0);
            opacity: 0.5;
        }
        100% {
            transform: scale(20, 20);
            opacity: 0;
        }
    }
        .trailer-button {
        background: linear-gradient(to right, #ff4b4b, #e63c3c);
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 15px;
        display: inline-block;
        text-decoration: none;
    }
    .trailer-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
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
                    st.markdown(
                      f'<a href="{trailer_url}" target="_blank" rel="noopener noreferrer" class="trailer-button">▶️ Watch Trailer</a>',
                     unsafe_allow_html=True
                           )

# Footer
st.markdown("""
<div class="footer">
    CineMatch AI • Powered by Streamlit • Movie data from TMDB
</div>
""", unsafe_allow_html=True)
