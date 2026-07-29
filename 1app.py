import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# =======================
# Custom CSS
# =======================
st.markdown("""
<style>

/* -------- Whole App Background -------- */
.stApp{
    background: linear-gradient(135deg, #87CEFA, #00BFFF);
}

/* -------- Sidebar -------- */
[data-testid="stSidebar"]{
    background: #5DADE2;
}

/* -------- Title -------- */
h1{
    color:white;
    text-align:center;
    font-weight:bold;
}

/* -------- Search Box -------- */
.stTextInput input{
    background-color:#E0F7FF !important;
    color:#000000 !important;
    border:2px solid #1E90FF !important;
    border-radius:10px !important;
}

/* -------- Search Box Label -------- */
.stTextInput label{
    color:white !important;
    font-weight:bold !important;
}

/* -------- Select Song Box -------- */
.stSelectbox div[data-baseweb="select"] > div{
    background-color:#E0F7FF !important;
    color:#000000 !important;
    border:2px solid #1E90FF !important;
    border-radius:10px !important;
}

/* -------- Select Song Label -------- */
.stSelectbox label{
    color:white !important;
    font-weight:bold !important;
}

/* -------- Dropdown List -------- */
div[role="listbox"]{
    background:#E0F7FF !important;
}

div[role="option"]{
    color:#000000 !important;
    background:#E0F7FF !important;
}

div[role="option"]:hover{
    background:#87CEFA !important;
}

/* -------- Buttons -------- */
.stButton>button{
    background:#1E90FF;
    color:white;
    border-radius:10px;
    font-size:18px;
    height:45px;
    width:100%;
}

.stButton>button:hover{
    background:#1565C0;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =======================
# Sidebar
# =======================
st.sidebar.title("🎧 Music Recommender")
st.sidebar.markdown("---")
st.sidebar.write("### Features")
st.sidebar.write("✅ Music Recommendation")
st.sidebar.write("✅ Dataset Statistics")
st.sidebar.write("✅ Language Distribution")
st.sidebar.write("✅ Search Songs")
st.sidebar.markdown("---")
st.sidebar.success("Made with ❤️ Streamlit")

# =======================
# Title
# =======================
st.title("🎵 Music Recommendation System")

# =======================
# Load Dataset
# =======================
df = pd.read_csv("final_music_data_updated.csv")

# =======================
# Dataset Statistics
# =======================
total_songs = len(df)
total_singers = df['artist'].nunique() if 'artist' in df.columns else "N/A"
total_languages = df['language'].nunique() if 'language' in df.columns else "N/A"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎵 Total Songs", total_songs)

with col2:
    st.metric("👨‍🎤 Total Artists", total_singers)

with col3:
    st.metric("🌍 Languages", total_languages)

st.markdown("---")


# =======================
# Top 5 Songs
# =======================
st.subheader("🎵 Top 5 Songs in Dataset")

top5 = df['song_name'].head(5)

for i, song in enumerate(top5, start=1):
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,#ff6a00,#ee0979);
            color:white;
            padding:15px;
            border-radius:10px;
            margin-bottom:10px;
            font-size:18px;
            font-weight:bold;
        ">
            🎵 {i}. {song}
        </div>
        """,
        unsafe_allow_html=True
    )
# =======================
# Generate Similarity Matrix
# =======================
if not os.path.exists("similarities.pkl"):
    if st.button("Generate Similarities"):
        cv = CountVectorizer(max_features=10000, stop_words='english')
        dtm = cv.fit_transform(df['tags'])
        similarities = cosine_similarity(dtm)

        pickle.dump(similarities, open("similarities.pkl", "wb"))
        st.success("Similarity Matrix Generated Successfully!")

# =======================
# Song List
# =======================
names = sorted(df['song_name'].unique())

# =======================
# Search Song
# =======================
search = st.text_input("🔍 Search Song") 


if search:
    names = [song for song in names if search.lower() in song.lower()]

# =======================
# Helper Functions
# =======================
def get_song_index(name):
    for i in df.index:
        if name == df.loc[i, 'song_name']:
            return i
    return -1

def get_song_name(i):
    if i >= len(df):
        return ""
    return df.loc[i, 'song_name']

# =======================
# Select Song
# =======================
name = st.selectbox("🎵 Select a Song", names)

# =======================
# Song Information
# =======================
song_data = df[df['song_name'] == name]

if not song_data.empty:
    st.info(f"""
🎵 **Song:** {song_data.iloc[0]['song_name']}

👨‍🎤 **Artist:** {song_data.iloc[0]['artist']}

🌍 **Language:** {song_data.iloc[0]['language']}
""")

# =======================
# Recommendation
# =======================
if st.button("🎶 Recommend Songs"):

    index = get_song_index(name)

    if index == -1:
        st.error("Song not found!")

    else:

        similarities = pickle.load(open("similarities.pkl", "rb"))

        similarity_index = list(enumerate(similarities[index]))

        similarity_index = sorted(
            similarity_index,
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("🎧 Recommended Songs")

        count = 1

        for i in similarity_index[1:6]:

            st.markdown(f"""
<div class="song-card">
<h4 style="color:#000000;">🎵 {count}. {get_song_name(i[0])}</h4>
</div>
""", unsafe_allow_html=True)

            count += 1

        st.markdown(
    "<h1 style='text-align:center;'>🎵Thank You🎵</h1>",
    unsafe_allow_html=True
)
        st.snow()
# =======================
# Footer
# =======================
st.markdown("---")
st.caption("🎵 Music Recommendation System | Developed using Streamlit ❤️")