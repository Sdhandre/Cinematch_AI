import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Page config
st.set_page_config(page_title="CineMatch AI", layout="wide")

# 2. Inject custom CSS for appearance and responsiveness
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #f0f0f0;
}
.stApp {
    background: linear-gradient(135deg, #121212 0%, #0a0a0a 100%);
}
/* Header */
.header-container {
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}
.header-container h1 {
    font-size: 2.5rem;
    color: #ff4b4b;
    margin-bottom: 0.3rem;
}
.header-container p {
    font-size: 1.1rem;
    margin: 0;
    padding: 0;
}
/* Filter container */
.filter-container {
    background: rgba(30, 30, 30, 0.7);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 1.5rem;
}
/* Movie card */
.movie-card {
    background: rgba(30, 30, 30, 0.8);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.movie-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    transform: translateY(-2px);
}
.title {
    color: #ff4b4b;
    font-weight: 700;
    font-size: 1.4rem;
    margin-bottom: 4px;
}
/* Tags */
.tag {
    display: inline-block;
    background: rgba(255, 75, 75, 0.2);
    padding: 4px 8px;
    border-radius: 4px;
    margin: 4px 6px 4px 0;
    font-size: 0.8em;
    color: #f0f0f0;
}
/* Buttons */
.stButton>button {
    background-color: #ff4b4b;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
    margin-bottom: 10px;
}
.stButton>button:hover {
    background-color: #e63c3c;
    transform: scale(1.02);
}
@media (max-width: 768px) {
    .header-container h1 {
        font-size: 2rem;
    }
    .header-container p {
        font-size: 1rem;
    }
    .title {
        font-size: 1.2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# 3. Load and preprocess data (cached)
@st.cache_data
def load_data():
    df_local = pd.read_csv("shortened_movie_dataset5.csv")
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

# 5. Header in main area
st.markdown(
    '<div class="header-container"><h1>🎬 CineMatch AI</h1><p>Discover your perfect movie match</p></div>',
    unsafe_allow_html=True
)

# 6. Define era mapping
ERA_OPTIONS = {
    "All Time": None,
    "2000-2010": (2000, 2010),
    "2010-2020": (2010, 2020),
    "2020-2024": (2020, 2024),
    "Classic (Pre-2000)": (None, 1999),
}

# 7. Filters at the top (no sidebar)
with st.container():
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    # Use expander if you want collapsible filters; can remove expander for always-visible
    with st.expander("🎯 Tell Us What You're Looking For", expanded=True):
        # First row: Mood & Genre
        col1, col2 = st.columns(2)
        with col1:
            selected_mood = st.selectbox(
                "Mood 😀",
                ["Happy", "Sad", "Thrilling", "Mind-Bending", "Romantic", "Nostalgic", "Inspired"],
                index=0
            )
        with col2:
            if 'genres' in df.columns:
                unique_genres = sorted({g.strip() for sub in df['genres'].dropna().str.split(',') for g in sub})
            else:
                unique_genres = []
            if unique_genres:
                default_idx = unique_genres.index('Drama') if 'Drama' in unique_genres else 0
                selected_genre = st.selectbox("Genre 😶‍🌫️", unique_genres, index=default_idx)
            else:
                selected_genre = ""
        # Second row: Language & Era
        col3, col4 = st.columns(2)
        with col3:
            if 'original_language' in df.columns:
                langs = sorted(df['original_language'].dropna().unique())
                selected_language = st.selectbox("Preferred language", ['All'] + langs, index=0)
            else:
                selected_language = 'All'
        with col4:
            selected_era_label = st.selectbox("Time period ⏲️", list(ERA_OPTIONS.keys()), index=0)
            era_range = ERA_OPTIONS[selected_era_label]
        # Third row: Pacing, Watching with, Ending type
        col5, col6, col7 = st.columns(3)
        with col5:
            selected_pace = st.selectbox(
                "Pacing 💨",
                ["Slow and Deep", "Medium and Balanced", "Fast and Action-packed"],
                index=1
            )
        with col6:
            watching_with = st.selectbox("Watching with 🧐", ["Alone", "Family", "Partner", "Friends"], index=0)
        with col7:
            selected_ending = st.selectbox(
                "Ending type 💀",
                ["Happy", "Realistic / Neutral", "Twist / Cliffhanger", "Sad", "Open to anything"],
                index=4
            )
    st.markdown('</div>', unsafe_allow_html=True)

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

# 11. "Find My Perfect Movie" button logic
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
    st.markdown("### Your Recommendations")
    # Buttons: Refresh, Previous Page, Next Page
    colr, colp1, colp2 = st.columns([1,1,1])
    with colr:
        if st.button("🔄 Refresh Recommendations"):
            top_pool = st.session_state['top_pool']
            sample_size = 10
            if len(top_pool) <= sample_size:
                st.session_state['random_sample'] = top_pool.copy()
            else:
                st.session_state['random_sample'] = random.sample(top_pool, sample_size)
            st.session_state['mode'] = 'random'
    with colp1:
        if st.button("⬅️ Previous Page"):
            st.session_state['mode'] = 'pagination'
            total = len(st.session_state['top_pool'])
            pages = max(1, (total + 9) // 10)
            st.session_state['page'] = (st.session_state['page'] - 1) % pages
    with colp2:
        if st.button("➡️ Next Page"):
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
                st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    path = row.get('poster_path') or row.get('poster_path')
                    if isinstance(path, str) and path.strip():
                        st.image(f"https://image.tmdb.org/t/p/w500{path}", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                with col2:
                    title = row.get("title") or row.get("name") or "Unknown Title"
                    st.markdown(f'<div class="title">{title}</div>', unsafe_allow_html=True)
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
                        st.markdown(f'<span class="tag">{str(row["genres"]).title()}</span>', unsafe_allow_html=True)
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
                                    background-color: #ff4b4b;
                                    color: white;
                                    padding: 8px 16px;
                                    border: none;
                                    border-radius: 6px;
                                    font-weight: 600;
                                    cursor: pointer;
                                    transition: all 0.2s ease;">
                                    ▶️ Watch Trailer
                                </button>
                            </a>
                            """,
                            unsafe_allow_html=True
                        )
                    


