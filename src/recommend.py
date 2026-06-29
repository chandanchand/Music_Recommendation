import joblib
import logging
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("Loading data...")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    df = joblib.load(BASE_DIR / "df_cleaned.pkl")
    tfidf_matrix = joblib.load(BASE_DIR / "tfidf_matrix.pkl")

    logging.info("Data loaded successfully.")

except Exception as e:
    logging.error(str(e))
    raise e


def recommend_songs(song_name, top_n=5):

    idx = df[df["song"].str.lower() == song_name.lower()].index

    if len(idx) == 0:
        return None

    idx = idx[0]

    # Calculate similarity only for selected song
    similarity = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity.argsort()[::-1][1:top_n+1]

    recommendations = (
        df.iloc[similar_indices][["artist", "song"]]
        .reset_index(drop=True)
    )

    recommendations.index += 1
    recommendations.index.name = "S.No."

    return recommendations
