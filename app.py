import streamlit as st
import pandas as pd
import os
import pickle
import faiss
import requests # For TMDB API calls
import numpy as np # Faiss might return numpy types

# --- Configuration & Constants ---
DATA_DIR = 'data'
FINAL_DIR = 'final'
MODEL_DIR = 'models'

DEMOGRAPHIC_CSV = os.path.join(DATA_DIR, FINAL_DIR, 'demographic_genres.csv')
MOVIES_CSV = os.path.join(DATA_DIR, 'prepared_movies.csv')
INDICES_PKL = os.path.join(DATA_DIR, FINAL_DIR,'indices.pkl')

MATRIX_PKL_ALL = os.path.join(DATA_DIR, FINAL_DIR,'all_matrix_dense.pkl')
FAISS_BIN_ALL = os.path.join(DATA_DIR, FINAL_DIR, 'all_faiss.bin')
MATRIX_PKL_THEME = os.path.join(DATA_DIR, FINAL_DIR,'theme_plot_matrix_dense.pkl')
FAISS_BIN_THEME = os.path.join(DATA_DIR, FINAL_DIR, 'theme_plot_faiss.bin')
MATRIX_PKL_PEOPLE = os.path.join(DATA_DIR, FINAL_DIR,'people_matrix_dense.pkl')
FAISS_BIN_PEOPLE = os.path.join(DATA_DIR, FINAL_DIR, 'people_faiss.bin')

SVD_PKL = os.path.join(DATA_DIR, MODEL_DIR, 'best_SVD_model.pkl')

TMDB_API_KEY = "7a238d6af7cb922d5ecf570d17419274"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342" 
DEFAULT_POSTER = "https://placehold.co/342x513/CCCCCC/000000?text=No+Poster" # Placeholder

# Top user IDs
TOP_USER_IDS = [8659, 37222, 39742, 45811, 70648, 98415, 98787, 141589, 165352, 172224]

# --- Data Loading (Cached) ---

@st.cache_data # Cache data loading
def load_dataframes():
    try:
        df_demographic = pd.read_csv(DEMOGRAPHIC_CSV)
        df_movies = pd.read_csv(MOVIES_CSV)
        # Safely evaluate genres list
        df_movies['genres'] = df_movies['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        df_movies['release_year'] = pd.to_datetime(df_movies['release_date'], errors='coerce').dt.year
        # Extract unique genres and years for filters
        all_genres = sorted(list(df_demographic['genre'].dropna().unique()))
        all_years = sorted(list(df_demographic['release_year'].dropna().astype(int).unique()), reverse=True)
        all_movie_titles = sorted(list(df_movies['title'].dropna().unique()))
        return df_demographic, df_movies, all_genres, all_years, all_movie_titles
    except FileNotFoundError as e:
        st.error(f"Error loading CSV data: {e}. Make sure files are in the '{DATA_DIR}' folder.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading or processing DataFrames: {e}")
        st.stop()


@st.cache_resource # Cache resource loading (models, index)
def load_models_and_index():
    try:
        with open(INDICES_PKL, 'rb') as f:
            indices = pickle.load(f)
        
        with open(MATRIX_PKL_ALL, 'rb') as f:
            combined_matrix_dense_all = pickle.load(f)
            # Ensure it's float32 for Faiss
            if combined_matrix_dense_all.dtype != np.float32:
                 combined_matrix_dense_all = combined_matrix_dense_all.astype(np.float32)
        faiss_index_all = faiss.read_index(FAISS_BIN_ALL)

        with open(MATRIX_PKL_THEME, 'rb') as f:
            combined_matrix_dense_theme = pickle.load(f)
            if combined_matrix_dense_theme.dtype != np.float32:
                 combined_matrix_dense_theme = combined_matrix_dense_theme.astype(np.float32)
        faiss_index_theme = faiss.read_index(FAISS_BIN_THEME)

        with open(MATRIX_PKL_PEOPLE, 'rb') as f:
            combined_matrix_dense_people = pickle.load(f)
            if combined_matrix_dense_people.dtype != np.float32:
                 combined_matrix_dense_people = combined_matrix_dense_people.astype(np.float32)
        faiss_index_people = faiss.read_index(FAISS_BIN_PEOPLE)

        with open(SVD_PKL, 'rb') as f:
            svd_model = pickle.load(f)

        # Check Faiss index dimension
        if faiss_index_all.d != combined_matrix_dense_all.shape[1]:
            st.warning(f"Warning: Faiss index dimension ({faiss_index_all.d}) doesn't match loaded matrix dimension ({combined_matrix_dense_all.shape[1]}). Ensure they were built together.")

        if faiss_index_theme.d != combined_matrix_dense_theme.shape[1]:
            st.warning(f"Warning: Faiss index dimension ({faiss_index_theme.d}) doesn't match loaded matrix dimension ({combined_matrix_dense_theme.shape[1]}). Ensure they were built together.")
        
        if faiss_index_people.d != combined_matrix_dense_people.shape[1]:
            st.warning(f"Warning: Faiss index dimension ({faiss_index_people.d}) doesn't match loaded matrix dimension ({combined_matrix_dense_people.shape[1]}). Ensure they were built together.")
        
        return indices, combined_matrix_dense_all, faiss_index_all, combined_matrix_dense_theme, faiss_index_theme, combined_matrix_dense_people, faiss_index_people, svd_model
    
    except FileNotFoundError as e:
        st.error(f"Error loading model/index file: {e}. Make sure files are in the '{DATA_DIR}' folder.")
        st.stop()
    
    except Exception as e:
        st.error(f"An error occurred loading models/index: {e}")
        st.stop()


# Load all data and models
df_demographic_genres, df_movies, ALL_GENRES, ALL_YEARS, ALL_MOVIE_TITLES = load_dataframes()
indices, combined_matrix_dense_all, faiss_index_all, combined_matrix_dense_theme, faiss_index_theme, combined_matrix_dense_people, faiss_index_people, svd = load_models_and_index()

# --- Helper Functions ---

# Function to fetch poster URL (with caching and error handling)
@st.cache_data
def fetch_poster(movie_id):
    # Ensure movie_id is a valid integer or can be converted to one
    try:
        valid_movie_id = int(movie_id)
    except (ValueError, TypeError):
        print(f"Invalid movie ID format: {movie_id}")
        return DEFAULT_POSTER

    url = f"https://api.themoviedb.org/3/movie/{valid_movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return POSTER_BASE_URL + poster_path
        else:
            # print(f"No poster path found for movie ID {valid_movie_id}")
            return DEFAULT_POSTER # Return placeholder if no poster path
    except requests.exceptions.HTTPError as http_err:
        # Specifically handle HTTP errors (like 404 Not Found)
        if response.status_code == 404:
            print(f"Movie ID {valid_movie_id} not found on TMDB.")
        else:
            print(f"HTTP error occurred for movie ID {valid_movie_id}: {http_err} - Status Code: {response.status_code}")
        return DEFAULT_POSTER
    except requests.exceptions.RequestException as e:
        print(f"API request failed for movie ID {valid_movie_id}: {e}")
        return DEFAULT_POSTER # Return placeholder on API error
    except Exception as e:
        print(f"Error fetching poster for movie ID {valid_movie_id}: {e}")
        return DEFAULT_POSTER


# Function to find movie index (modified for Streamlit)
def find_movie_index(title, indices_map):
    if title in indices_map:
        idx = indices_map[title]
        # Handle scalar vs Series return from pickle
        return idx.iloc[0] if isinstance(idx, pd.Series) else idx
    # No close matches here, rely on selectbox ensuring title exists
    return None

# --- Recommendation Model Functions (Adapted from user script) ---

def popularity_model(df, genres=None, release_years=None):
    df_filtered = df.copy() # Work on a copy
    # Filter by genres if specified (and not empty)
    if genres:
        # Group by movie ID or title and filter movies that contain all selected genres
        df_filtered = (
            df_filtered.groupby('title')  # Group by movie title (or 'id' if needed)
            .filter(lambda group: set(genres).issubset(set(group['genre'])))
        )

    # Filter by release years if specified (and not empty)
    if release_years:
        df_filtered = df_filtered[df_filtered['release_year'].isin(release_years)]

    if df_filtered.empty:
        return pd.DataFrame(columns=['title', 'vote_count', 'vote_average', 'score', 'id']) # Return empty with 'id'

    # Keep relevant columns and drop duplicates based on 'id'
    df_filtered = df_filtered[['id', 'title', 'vote_count', 'vote_average', 'release_year']].drop_duplicates(subset=['id'])

    # Calculate C and m based on the *filtered* data
    vote_averages = df_filtered['vote_average'].dropna()
    vote_counts = df_filtered['vote_count'].dropna()

    if vote_averages.empty or vote_counts.empty or len(vote_counts) == 0: # Check if empty after dropna
         return pd.DataFrame(columns=['title', 'vote_count', 'vote_average', 'score', 'id'])

    C = vote_averages.mean()
    # Ensure quantile calculation doesn't fail on small filtered sets
    # Use max(1, ...) to avoid m=0 if min is 0
    m = vote_counts.quantile(0.9) if len(vote_counts) >= 10 else max(1, vote_counts.min())


    # Filter movies qualified for the chart
    qualified = df_filtered[df_filtered['vote_count'] >= m].copy()

    if qualified.empty:
        return pd.DataFrame(columns=['title', 'vote_count', 'vote_average', 'score', 'id'])

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Avoid division by zero
        denominator = v + m
        if denominator == 0:
            return C # Return mean if denominator is zero
        return (v / denominator * R) + (m / denominator * C)

    qualified['score'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('score', ascending=False)

    return qualified[['title', 'vote_count', 'vote_average', 'score', 'id']].head(10)


def content_model(title, matrix_dense, faiss_index, n=10):
    idx = find_movie_index(title, indices)
    # Check if index is valid and within the bounds of the dense matrix
    if idx is None or not isinstance(idx, (int, np.integer)) or idx < 0 or idx >=matrix_dense.shape[0]:
        st.error(f"Could not find a valid index for movie '{title}'.")
        return pd.DataFrame(columns=['title', 'id', 'similarity']) # Return empty DataFrame

    query_vector = matrix_dense[int(idx)].reshape(1, -1)
    # Vectors in index are already normalized

    try:
        distances, movie_indices = faiss_index.search(query_vector, n + 1)
    except Exception as e:
        st.error(f"Faiss search failed: {e}")
        return pd.DataFrame(columns=['title', 'id', 'similarity'])

    # Filter out invalid indices (-1 can be returned by Faiss) and ensure bounds
    valid_mask = (movie_indices[0] != -1) & (movie_indices[0] < len(df_movies))
    movie_indices = movie_indices[0][valid_mask]
    distances = distances[0][valid_mask]

    # Exclude self-match if present
    self_mask = (movie_indices != int(idx))
    movie_indices = movie_indices[self_mask]
    similarity_scores = distances[self_mask]

    # Take top N after excluding self
    movie_indices = movie_indices[:n]
    similarity_scores = similarity_scores[:n]

    similarity_scores = [round(score, 2) for score in similarity_scores]

    # Ensure indices are valid for iloc
    valid_iloc_indices = [i for i in movie_indices if i < len(df_movies)]
    if not valid_iloc_indices:
        return pd.DataFrame(columns=['title', 'id', 'similarity'])

    results_df = df_movies.iloc[valid_iloc_indices][['title', 'id']].copy()
    # Ensure similarity_scores aligns with the potentially filtered results_df
    results_df['similarity'] = similarity_scores[:len(results_df)]


    return results_df

def get_hybrid_recommendations(user_id, title, matrix_dense, faiss_index, n=10):
    idx = find_movie_index(title, indices)
    # Check if index is valid and within the bounds of the dense matrix
    if idx is None or not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= matrix_dense.shape[0]:
        st.error(f"Could not find a valid index for movie '{title}'.")
        return pd.DataFrame(columns=['title', 'est_rating', 'id']) # Return empty DataFrame

    query_vector = matrix_dense[int(idx)].reshape(1, -1)
    # Vectors in index are already normalized

    try:
         # Increase candidate count slightly for more robust filtering later
        distances, movie_indices = faiss_index.search(query_vector, 25 + 1) # Get 25 candidates initially
    except Exception as e:
        st.error(f"Faiss search failed for hybrid: {e}")
        return pd.DataFrame(columns=['title', 'est_rating', 'id'])

    # Filter out invalid indices (-1 can be returned by Faiss) and ensure bounds
    valid_mask = (movie_indices[0] != -1) & (movie_indices[0] < len(df_movies))
    movie_indices = movie_indices[0][valid_mask]

    # Exclude self-match if present
    self_mask = (movie_indices != int(idx))
    movie_indices = movie_indices[self_mask]

    # Take top N candidates *after* excluding self
    movie_indices = movie_indices[:25] # Limit to 25 candidates

    if len(movie_indices) == 0:
         st.warning("No similar movies found after initial filtering.")
         return pd.DataFrame(columns=['title', 'est_rating', 'id'])

    # Get candidate movie details ('id' is tmdbId here)
    candidates = df_movies.iloc[movie_indices][['title', 'id']].copy()

    # Predict ratings using SVD model
    predictions = []
    known_user = svd.trainset.knows_user(svd.trainset.to_inner_uid(user_id)) if hasattr(svd, 'trainset') else True # Check if user is known

    for tmdb_id in candidates['id']:
        # Check if SVD model knows this item before predicting
        known_item = False
        try:
             # This assumes SVD was trained with tmdb_id as item ID
             inner_iid = svd.trainset.to_inner_iid(tmdb_id)
             known_item = True
        except ValueError:
             # print(f"Item ID {tmdb_id} not in the SVD training set.")
             known_item = False # Item not in training set

        if known_user and known_item:
            pred = svd.predict(user_id, tmdb_id)
            predictions.append({'id': tmdb_id, 'est_rating': pred.est})
        # else: # Optionally handle cases where user/item is unknown
            # print(f"Skipping prediction for unknown user {user_id} or item {tmdb_id}")


    if not predictions:
         st.warning("No SVD ratings could be estimated for the similar movies (possibly unknown user/items).")
         return pd.DataFrame(columns=['title', 'est_rating', 'id'])

    pred_df = pd.DataFrame(predictions)
    # Merge predictions back to candidates based on 'id' (tmdb_id)
    candidates = candidates.merge(pred_df, on='id', how='inner') # Use inner merge to keep only predicted movies
    candidates = candidates.dropna(subset=['est_rating']) # Should not be needed after inner merge

    # Sort by estimated rating and return top N
    return candidates.sort_values('est_rating', ascending=False)[['title', 'est_rating', 'id']].head(n)

# --- Display Function ---
def display_recommendations(df, score_col=None):
    """Displays movie recommendations with posters in columns."""
    if df is None or df.empty:
        st.write("No recommendations found for the selected criteria.")
        return

    # Ensure 'id' column exists for poster fetching
    if 'id' not in df.columns:
        st.error("Cannot display posters: 'id' column missing from recommendations.")
        st.dataframe(df)  # Display dataframe without posters
        return

    # Define number of columns dynamically, e.g., based on screen width or fixed
    num_cols = 5
    cols = st.columns(num_cols)
    col_idx = 0

    # Ensure unique rows if duplicates somehow occurred
    df_display = df.drop_duplicates(subset=['id']).head(10)  # Display max 10

    for index, row in df_display.iterrows():
        with cols[col_idx % num_cols]:  # Cycle through columns
            st.markdown(f"**{row['title']}**")
            # Fetch poster using the 'id' (tmdbId)
            poster_url = fetch_poster(row['id'])
            st.image(poster_url, width='stretch')  # Updated parameter

            # Display score if available
            if score_col and score_col in row and not pd.isna(row[score_col]):
                st.write(f"{score_col.replace('_', ' ').title()}: {row[score_col]:.2f}")

            # Expander for details
            with st.expander("Details"):
                try:
                    # Fetch full movie details from df_movies using 'id'
                    # Use .loc for safer indexing
                    movie_details_row = df_movies.loc[df_movies['id'] == row['id']]
                    if not movie_details_row.empty:
                        movie_details = movie_details_row.iloc[0]
                        year_val = movie_details.get('release_year')
                        genres_val = movie_details.get('genres')
                        vote_avg_val = movie_details.get('vote_average')
                        vote_count_val = movie_details.get('vote_count')
                        overview_val = movie_details.get('overview')

                        st.markdown(f"**Year:** {int(year_val)}" if pd.notna(year_val) else "N/A")
                        st.markdown(f"**Genres:** {', '.join(genres_val) if genres_val else 'N/A'}")
                        st.markdown(f"**Rating:** {vote_avg_val:.1f}/10 ({int(vote_count_val)} votes)" if pd.notna(vote_avg_val) and pd.notna(vote_count_val) else "N/A")
                        st.caption(f"**Overview:** {overview_val}" if pd.notna(overview_val) and overview_val else "No overview available.")
                    else:
                        st.write("Details not found in main movie dataframe.")
                except Exception as e:
                    st.write(f"Error displaying details: {e}")

            col_idx += 1


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("🎬 Your Personal Movie Recommender")
st.markdown("Discover movies based on popularity, content similarity, or personalized recommendations!")

# # Add custom CSS for styling tabs
# st.markdown(
#     """
#     <style>
#     /* Center the tabs */
#     .stTabs [role="tablist"] {
#         justify-content: center;
#     }

#     /* Style the tabs */
#     .stTabs [role="tab"] {
#         font-size: 18px;
#         font-weight: bold;
#         color: #4CAF50; /* Green color */
#         border: 2px solid #4CAF50;
#         border-radius: 10px;
#         padding: 10px;
#         margin: 0 5px;
#         background-color: #f9f9f9;
#     }

#     /* Highlight the active tab */
#     .stTabs [role="tab"][aria-selected="true"] {
#         background-color: #4CAF50;
#         color: white;
#     }

#     /* Hover effect for tabs */
#     .stTabs [role="tab"]:hover {
#         background-color: #e8f5e9;
#         color: #4CAF50;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Create tabs for each section
tab1, tab2, tab3 = st.tabs(["Popular Movies", "Find Similar Movies", "Personalized Picks"])

# --- Section 1: Popularity-Based ---
with tab1:
    st.header("Crowd Favorites")
    st.markdown("Find top-rated movies, optionally filtered by genre and release year.")

    col1_pop, col2_pop = st.columns([1, 1])
    with col1_pop:
        selected_genres = st.multiselect("Select Genres (leave empty for all):", ALL_GENRES)
        #st.write("Select Genres (check all that apply):")
        #selected_genres = [genre for genre in ALL_GENRES if st.checkbox(genre, key=f"{genre}")]
    with col2_pop:
        selected_years = st.multiselect("Select Release Years (leave empty for all):", ALL_YEARS)

    # Use context manager for spinner
    with st.spinner("Finding popular movies..."):
        popularity_recs = popularity_model(df_demographic_genres, genres=selected_genres or None, release_years=selected_years or None)

    display_recommendations(popularity_recs, score_col='score')

# --- Section 2: Content-Based---
with tab2:
    st.header("Find Movies Like...")
    st.markdown("Select a movie you like to get recommendations.")

    col1_rec = st.columns(1)[0]  # Single column for the movie title selectbox
    with col1_rec:
        # Use selectbox for existing titles - ensures the title is valid
        selected_movie_title = st.selectbox(
            "Select a Movie Title:",
            options=ALL_MOVIE_TITLES,
            index=ALL_MOVIE_TITLES.index("Interstellar") if "Interstellar" in ALL_MOVIE_TITLES else 0,  # Default selection
            key="content_based_selectbox"  # Unique key
        )

    # Content-Based Recommendations Section
    if st.button(f"Get Recommendations based on '{selected_movie_title}'", key="content_based_button"):
        # st.markdown("---")
        # --- Content-Based Output ALL ---
        st.subheader(f"Top Recommendations for '{selected_movie_title}'")
        with st.spinner("Finding similar movies..."):
            content_recs_01 = content_model(selected_movie_title, combined_matrix_dense_all, faiss_index_all, n=10)
        display_recommendations(content_recs_01, score_col='similarity')

        # --- Content-Based Output THEMES + PLOT ---
        st.subheader(f"Similar Stories and Themes for '{selected_movie_title}'")
        with st.spinner("Finding similar movies..."):
            content_recs_02 = content_model(selected_movie_title, combined_matrix_dense_theme, faiss_index_theme, n=10)
        display_recommendations(content_recs_02, score_col='similarity')

        # --- Content-Based Output People ---
        st.subheader(f"From the Cast and Crew of '{selected_movie_title}'")
        with st.spinner("Finding similar movies..."):
            content_recs_03 = content_model(selected_movie_title, combined_matrix_dense_people, faiss_index_people, n=10)
        display_recommendations(content_recs_03, score_col='similarity')


# --- Section 3: Hybrid ---
with tab3:
    st.header("Tailored Recommendations")
    st.markdown("Recommendations based on a movie you like, personalized for your user profile.")

    col1_per, col2_per = st.columns([2, 1])
    with col1_per:
        # Use selectbox for existing titles - ensures the title is valid
        selected_movie_title_ = st.selectbox(
            "Select a Movie Title:",
            options=ALL_MOVIE_TITLES,
            index=ALL_MOVIE_TITLES.index("Interstellar") if "Interstellar" in ALL_MOVIE_TITLES else 0,  # Default selection
            key="hybrid_selectbox_movie"  # Unique key
        )
    with col2_per:
        selected_user_id = st.selectbox(
            "Select User Profile (Top 10 Raters):",
            options=TOP_USER_IDS,
            index=0,
            key="hybrid_selectbox_user"  # Unique key
        )


    # Hybrid Recommendations Section
    if st.button(f"Get Recommendations based on '{selected_movie_title_}'", key="hybrid_button"):
        # st.markdown("---")
        # --- Hybrid Output ---
        st.subheader(f"Personalized Picks for User {selected_user_id} (Based on '{selected_movie_title_}')")
        with st.spinner(f"Calculating personalized recommendations for User {selected_user_id}..."):
            hybrid_recs = get_hybrid_recommendations(selected_user_id, selected_movie_title_, combined_matrix_dense_all, faiss_index_all, n=10)
        display_recommendations(hybrid_recs, score_col='est_rating')

st.markdown("---")
st.caption("Feel free to contact me in my [LinkedIn](https://www.linkedin.com/in/mary-nathalie-dela-cruz/) profile.")

st.caption("Movie data and Ratings data from TMDB (1990-2020). User profiles based on top raters.")
