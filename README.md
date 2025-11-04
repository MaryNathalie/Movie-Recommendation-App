# Movie Recommendation Application

This project is a comprehensive movie recommendation system built, featuring a Streamlit web application that provides recommendations through three different models: Demographic, Content-Based, and Hybrid filtering.

## 1\. Project Overview

This application provides users with multiple ways to discover movies:

  * **Popular Movies (Demographic Filtering):** A non-personalized "Top Movies" chart. It uses the IMDB weighted rating formula (based on `vote_average` and `vote_count`) to rank movies. This list can be filtered by **Genre** and **Release Year**.
  * **Find Similar Movies (Content-Based Filtering):** Recommends movies based on their similarity to a user-selected title. This model is broken down into three types:
    1.  **Plot + All Metadata:** Similarity based on plot overview, cast, crew, keywords, and genres.
    2.  **Plot + Themes:** Similarity based on plot overview, genres, and keywords.
    3.  **People:** Similarity based on cast, directors, and writers.
  * **Personalized Picks (Hybrid Filtering):** A two-stage model that combines content-based filtering with collaborative filtering. For a selected user and movie:
    1.  **Candidate Generation:** It first finds a pool of content-similar movies (using the Plot + All Metadata model).
    2.  **Personalized Re-ranking:** It then uses a **Singular Value Decomposition (SVD)** collaborative filtering model to predict how the selected user would rate those similar movies, re-ranking them based on the highest-predicted score.

The project leverages different libraries, including **Pandas** for data manipulation, **Scikit-learn** (`CountVectorizer`) and **SentenceTransformers** for feature extraction, **Faiss** for high-speed similarity search, **Surprise** (SVD) for collaborative filtering, and **Optuna** for hyperparameter tuning. The entire system is served via a **Streamlit** web application.

## 2\. Running the Application

### a. Prerequisites

  * Python (3.9+ recommended)
  * `pip` (Python package installer)
  * A **TMDB API Key** for fetching movie posters. You can get one by creating an account and registering for an API key on their [Getting Started](https://developer.themoviedb.org/docs/getting-started) page.

### b. Setup & Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/movie-recommendation-app.git
    cd movie-recommendation-app
    ```

2.  **Set up Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Key libraries include: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `faiss-cpu`, `sentence-transformers`, `surprise`, `optuna`, `requests`.*

4.  **Generate Model and Data Files:**
    The Streamlit app (`app.py`) depends on pre-computed data and models. You must run the Jupyter Notebooks in numerical order to generate these files.

      * `02_data_preparation.ipynb`: Creates `data/prepared_movies.csv` and `data/prepared_ratings.csv`.
      * `03_recommendation_demographic.ipynb`: Creates `data/final/demographic_genres.csv`.
      * `04_recommendation_content_plot.ipynb`: Creates `data/overview_embedding_matrix.pkl` and `data/overview_indices.pkl`.
      * `05_recommendation_content_metadata.ipynb`: Creates `data/metadata_*.pkl` files.
      * `06_recommendation_collaborative.ipynb`: Creates `data/models/best_SVD_model.pkl`.
      * `08_build_final_files.ipynb`: Creates the final matrices and Faiss indices in `data/final/`.

5.  **Add TMDB API Key:**
    Open the `app.py` file and replace the placeholder value for `TMDB_API_KEY`:

    ```python
    # Near the top of app.py
    TMDB_API_KEY = "YOUR_API_KEY_GOES_HERE" 
    ```

6.  **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

## 3\. Dataset Summary

The project uses "The Movies Dataset" from Kaggle, which includes:

  * `movies_metadata.csv`: Detailed movie information (title, overview, genres, release date, etc.) for \~45,000 movies.
  * `credits.csv`: Cast and crew information (in JSON format) for all movies.
  * `keywords.csv`: Movie keywords (in JSON format).
  * `ratings.csv`: Over 26 million user ratings from 270,000 users.

**Data Preparation (`02_data_preparation.ipynb`):**
The raw data was heavily cleaned and filtered to create a robust working dataset. Key steps included:

  * Merging all data sources on the movie `id`.
  * Filtering for **English-language** movies only.
  * Filtering for movies with a "Released" status, released **after 1990**.
  * Removing movies with no overview.
  * Extracting, cleaning, and limiting cast/crew lists (e.g., top 3 cast, key directors/writers).

This resulted in a final dataset of **\~17,000 movies** and **\~18 million ratings** used for the models.

## 4\. Design Decisions

This project involved several key design decisions to balance performance, accuracy, and user experience:

  * **Multiple Recommendation Strategies:** We implemented three distinct models (Demographic, Content, Hybrid) because no single recommender is best for all scenarios. This provides a "cold start" solution (Demographic), a simple "find similar" tool (Content), and a personalized option (Hybrid).
  * **Feature Engineering (Content):**
      * **Plot Semantics:** Used **`SentenceTransformer`** (`all-mpnet-base-v2`) to create dense 768-dimension embeddings from movie overviews. This captures semantic meaning ("What is the movie *about*?") better than traditional methods like TF-IDF.
      * **Metadata Tokens:** Used **`CountVectorizer`** for discrete metadata (cast, crew, keywords, genres). Names were concatenated (e.g., `tomhanks`) to be treated as single, distinct tokens. Keywords were stemmed to reduce dimensionality.
      * **Hybrid Content Matrix:** The "All" and "Theme" content models stack the sparse metadata matrix and the dense plot matrix (`scipy.sparse.hstack`) into a single feature matrix. This combines the "who" and "what" of a movie for a richer similarity signal.
  * **Collaborative Filtering Model:**
      * **SVD (Surprise):** Chose Singular Value Decomposition (SVD) from the `surprise` library as it's a powerful and well-established matrix factorization algorithm for collaborative filtering.
      * **Hyperparameter Tuning:** Used **Optuna** to automatically search for the best hyperparameters (`n_factors`, `n_epochs`, `lr_all`, `reg_all`) for the SVD model, optimizing for the lowest **RMSE** (Root Mean Squared Error) on a 3-fold cross-validation set.
      * **Model Evaluation:** The final SVD model was evaluated on a held-out test set (20% of the ratings data). The key metrics were:
          * **RMSE (Root Mean Squared Error):** 0.8177
          * **MAE (Mean Absolute Error):** 0.6199
          * **FCP (Fraction of Concordant Pairs):** 0.7182
  * **Performance (Faiss):**
      * Calculating cosine similarity on-the-fly for 17,000 movies is too slow for a web app.
      * We adopted **Faiss** (from Facebook AI) to build an in-memory `IndexFlatIP`.
      * Matrices were L2-normalized, allowing us to use the high-speed Inner Product (IP) search, which becomes mathematically equivalent to cosine similarity. This provides near-instantaneous similarity search.
  * **Hybrid Model (Two-Stage):**
      * Predicting SVD scores for all 17,000 movies for a user is too slow.
      * We implemented a two-stage "candidate generation & re-ranking" pipeline:
    <!-- end list -->
    1.  **Stage 1 (Content):** Faiss rapidly selects the top 25 movies most similar to the user's input movie.
    2.  **Stage 2 (Collaborative):** The SVD model predicts a rating for *only those 25 candidates* based on the selected user's profile.
    <!-- end list -->
      * This provides a highly relevant, personalized, and fast recommendation.

## 5\. Use Case and Examples

The Streamlit application (`app.py`) provides a simple UI for each model:

  * **Use Case 1: Popular Movies (Demographic)**

      * **User:** "I want to find a popular movie, maybe a Comedy or Romance from the 1990s."
      * **Action:** The user navigates to the "Popular Movies" tab, selects "Comedy" and "Romance" from the genre multiselect, and "1995", "1996", etc., from the year multiselect.
      * **Result:** The app displays the top 10 movies that match all criteria, ranked by their weighted score.

  * **Use Case 2: Find Similar Movies (Content-Based)**

      * **User:** "I just watched *Inception* and loved it. I want to see more movies like it."
      * **Action:** The user goes to the "Find Similar Movies" tab and selects "Inception" from the dropdown.
      * **Result:** The app displays three lists:
        1.  **Top Recommendations:** Movies similar in plot, themes, and people (e.g., *The Dark Knight*, *The Martian*).
        2.  **Similar Stories and Themes:** Movies with similar plots and keywords (e.g., *Apollo 13*, *Love*).
        3.  **From the Cast and Crew:** Movies sharing key actors or directors (e.g., *The Dark Knight Rises*, *The Prestige*).

  * **Use Case 3: Personalized Picks (Hybrid)**

      * **User:** "I'm User 45811 (a power user), and I also liked *Inception*. What would *I* like most?"
      * **Action:** The user goes to the "Personalized Picks" tab, selects "Inception" as the movie and "45811" as the user profile.
      * **Result:** The app finds movies similar to *Inception* (like *The Dark Knight Rises*, *The Martian*, *Passengers*), but re-ranks them based on User 45811's past rating behavior, showing the movies they are *most likely* to enjoy personally.

## 6\. Challenges Faced and Solutions Implemented

  * **Challenge:** The raw datasets were large (26M+ ratings), disconnected, and contained messy, unstructured data (e.g., JSON as strings).

      * **Solution:** A dedicated data preparation notebook (`02_data_preparation.ipynb`) was created to parse, clean, filter, and merge all sources into analysis-ready files (`prepared_movies.csv`, `prepared_ratings.csv`).

  * **Challenge:** Real-time content-based similarity search (e.g., `cosine_similarity(matrix, matrix)`) on a 17,000-item matrix is computationally infeasible for an interactive app.

      * **Solution:** We pre-computed all feature matrices and indexed them using **Faiss** (`IndexFlatIP`). This allows for sub-second similarity lookups by the Streamlit app.

  * **Challenge:** Combining semantic plot information (from `SentenceTransformer`) with discrete metadata (from `CountVectorizer`).

      * **Solution:** We stacked the sparse and dense matrices (`scipy.sparse.hstack`) into a single, wide feature matrix. This unified matrix was then indexed by Faiss, allowing the similarity search to consider both factors simultaneously.

  * **Challenge:** The SVD model from `surprise` required extensive tuning to find the best hyperparameters.

      * **Solution:** We integrated **Optuna** (`06_recommendation_collaborative.ipynb`) to automate the hyperparameter search, which tested 20 trial combinations to find the parameters that minimized RMSE.

  * **Challenge:** A purely collaborative filter (SVD) cannot recommend items to new users or items with no ratings (cold start).

      * **Solution:** We implemented a **hybrid model**. The content-based component finds similar items (which works for any movie), and the SVD component personalizes that list if the user is known. For unknown users/items, the app can default back to the pure content-based or demographic results.

## 7\. Future Improvements

  * **Advanced Hybrid Models:** Implement a deeply integrated hybrid model like two stage recommendation systems.
  * **Scalability:** For a larger dataset, the current in-memory (Pandas/Faiss) approach would fail. This could be scaled by:
      * Moving data processing to **Spark**.
      * Storing vectors in a dedicated vector database (e.g., **Milvus, Pinecone, or Weaviate**).
      * Serving the SVD model from a more robust prediction service.
  * **Full User Lifecycle:** Expand the UI to allow new users to register and rate movies. This would require a "cold start" strategy (e.g., showing demographic recommendations until they have rated \~10 movies) and periodic re-training of the SVD model.
  * **API-driven Architecture:** Separate the backend (model prediction) from the frontend (Streamlit) by building a **FastAPI** to serve the recommendations. The Streamlit app would then just call this API, making the system more modular and robust.
  * **Expanded Evaluation:** Train more models and add more recommendation-specific evaluation metrics beyond RMSE/MAE, such as **Precision@k**, **Recall@k**, and **Mean Average Precision (MAP)**, to better judge the quality of the ranked recommendation lists.
