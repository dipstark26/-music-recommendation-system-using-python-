import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss

# -------------------------- Streamlit Page Setup -------------------------- #
st.set_page_config(page_title="Spotify Music App & Recommendation System", layout="wide")
st.title("Spotify Music App & Recommendation System")
st.markdown(
    """
    This app allows you to search for tracks on Spotify as well as get song recommendations based on our dataset.
    """
)

# ------------------------- Spotify Authentication ------------------------- #
# Hardcoded credentials from you:
spotify_client_id = "9fa853af7e244c83857bcad982f34f20"
spotify_client_secret = "912a5c12b6454a67bc068fabcfb9106b"

try:
    client_credentials_manager = SpotifyClientCredentials(
        client_id=spotify_client_id, client_secret=spotify_client_secret
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    st.success("Spotify authentication successful!")
except Exception as e:
    st.error(f"Spotify authentication error: {e}")

# ---------------------- App Functionalities (Tabs) ----------------------- #
tabs = st.tabs(["Spotify Search", "Song Recommendations"])

# ----------------------- Tab 1: Spotify Search ---------------------------- #
with tabs[0]:
    st.header("Spotify Search")
    search_query = st.text_input("Enter a track name to search on Spotify", "")
    
    if search_query:
        try:
            results = sp.search(q=search_query, type="track", limit=10)
            tracks = results.get("tracks", {}).get("items", [])
            
            if not tracks:
                st.info("No tracks found. Please try a different query.")
            else:
                for track in tracks:
                    track_name = track.get("name", "Unknown")
                    artist_names = [artist["name"] for artist in track.get("artists", [])]
                    album_name = track.get("album", {}).get("name", "Unknown")
                    track_url = track.get("external_urls", {}).get("spotify", "#")
                    
                    album_images = track.get("album", {}).get("images", [])
                    if album_images:
                        image_url = album_images[0].get("url")
                        st.image(image_url, width=200)
                    
                    st.markdown(f"**Track:** {track_name}")
                    st.markdown(f"**Artist(s):** {', '.join(artist_names)}")
                    st.markdown(f"**Album:** {album_name}")
                    st.markdown(f"[Listen on Spotify]({track_url})")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error during Spotify search: {e}")

# ------------------- Tab 2: Song Recommendations -------------------------- #
with tabs[1]:
    st.header("Song Recommendations from Our Dataset")
    
    @st.cache_data(show_spinner=False)
    def load_dataset(filepath="spotify_millsongdata.csv"):
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    
    songs = load_dataset()
    if songs is not None:
        required_columns = ["artist", "song", "link", "text"]
        if not all(col in songs.columns for col in required_columns):
            st.error(f"Dataset is missing one of the required columns: {required_columns}")
        else:
            songs["text"] = songs["text"].fillna("")
            songs["display_name"] = songs["artist"] + " - " + songs["song"]
            
            selected_song = st.selectbox("Select a song", songs["display_name"].unique())
            song_idx = songs[songs["display_name"] == selected_song].index[0]
            
            @st.cache_data(show_spinner=False)
            def build_embeddings(text_series, n_components=100):
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(text_series)
                n_comp = min(n_components, tfidf_matrix.shape[1] - 1)
                svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
                svd_matrix = svd_model.fit_transform(tfidf_matrix)
                return svd_matrix
            
            svd_matrix = build_embeddings(songs["text"])
            
            @st.cache_resource(show_spinner=False)
            def build_faiss_index(embeddings):
                dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(embeddings.astype(np.float32))
                return index
            
            index = build_faiss_index(svd_matrix)
            
            def get_similar_songs(song_idx, svd_matrix, index, top_k=5):
                query_vector = svd_matrix[song_idx : song_idx + 1].astype(np.float32)
                distances, indices = index.search(query_vector, top_k + 1)
                similar_indices = [i for i in indices[0] if i != song_idx][:top_k]
                return similar_indices, distances[0]
            
            similar_indices, distances = get_similar_songs(song_idx, svd_matrix, index, top_k=5)
            st.markdown(f"### Recommendations for **{selected_song}**:")
            
            if similar_indices:
                # Create a row of columns (one column per recommendation)
                rec_cols = st.columns(len(similar_indices))
                for col, (idx, dist) in zip(rec_cols, zip(similar_indices, distances)):
                    with col:
                        rec_song = songs.iloc[idx]
                        rec_display = f"{rec_song.get('artist', 'Unknown')} - {rec_song.get('song', 'Unknown')}"
                        
                        # Fetch album image from Spotify based on artist and song name
                        query = f"{rec_song.get('artist', '')} {rec_song.get('song', '')}"
                        try:
                            search_results = sp.search(q=query, type="track", limit=1)
                            items = search_results.get('tracks', {}).get('items', [])
                            if items:
                                track_item = items[0]
                                album_images = track_item.get("album", {}).get("images", [])
                                if album_images:
                                    image_url = album_images[0].get("url")
                                    st.image(image_url, width=200)
                                else:
                                    st.write("No image available")
                            else:
                                st.write("No image found")
                        except Exception as e:
                            st.write("Error fetching image")
                        
                        st.markdown(f"**{rec_display}**")
                        st.markdown(f"(Distance: {dist:.4f})")
            else:
                st.info("No similar songs found.")
    else:
        st.info("Dataset could not be loaded or is empty.")
