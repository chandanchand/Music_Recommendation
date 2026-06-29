import streamlit as st
from PIL import Image
from recommend import df, recommend_songs
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
with open(BASE_DIR / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.title("🎵 Music AI")

    st.markdown("---")

    st.write("""
Discover similar songs using **Natural Language Processing**
and **Machine Learning**.
""")

    st.markdown("---")

    st.markdown("### Tech Stack")

    st.markdown("""
- Python
- Streamlit
- Pandas
- Scikit-Learn
- TF-IDF
- Cosine Similarity
""")

# --------------------------------------------------
# Logo
# --------------------------------------------------
logo = Image.open(BASE_DIR / "assets" / "logo.png")

# --------------------------------------------------
# Header
# --------------------------------------------------
col1, col2 = st.columns([1,5])

with col1:
    st.image(logo, width=90)

with col2:

    st.title("🎵 Music Recommendation System")

    st.caption("Discover songs using NLP & Machine Learning")

st.write("")

# --------------------------------------------------
# Metrics
# --------------------------------------------------
m1, m2, m3 = st.columns(3)

m1.metric("🎵 Songs", len(df))
m2.metric("🤖 Model", "TF-IDF")
m3.metric("⚡ Engine", "Cosine Similarity")

st.write("")

# --------------------------------------------------
# Search
# --------------------------------------------------
song_list = sorted(df["song"].dropna().unique())

selected_song = st.selectbox(
    "🎧 Search Song",
    song_list
)

st.write("")

# --------------------------------------------------
# Recommendation
# --------------------------------------------------
if st.button("🎶 Recommend Songs", use_container_width=True):

    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song)

    st.divider()

    st.subheader("🎵 Top Recommendations")

    if recommendations is None:

        st.error("Song not found!")

    else:

        col1, col2 = st.columns(2)

        cols = [col1, col2]

        for i, (_, row) in enumerate(recommendations.iterrows()):

            with cols[i % 2]:

                with st.container(border=True):

                    st.subheader(f"🎵 {row['song']}")

                    st.caption(f"👤 {row['artist']}")

                    st.write(
                        "Recommended using **TF-IDF** & **Cosine Similarity**."
                    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption("🎵 Music Recommendation System | Built with Streamlit")