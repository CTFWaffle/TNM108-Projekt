import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures

@st.cache_data
def load_data():
    # Load the dataset in chunks
    chunk_size = 1000
    chunks = pd.read_csv('E:/Programmering/TNM108-Projekt/Datasets/beer_reviews_cleaned.csv', chunksize=chunk_size)

    # Initialize empty DataFrame to store processed chunks
    beers = pd.DataFrame()

    # Process each chunk
    for chunk in chunks:
        beers = pd.concat([beers, chunk], ignore_index=True)

    return beers

@st.cache_data
def preprocess_data(beers):
    # Combine text columns to create a 'beer_description'
    beers['beer_description'] = beers['beer_style'] + ' ' + beers['beer_name']

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the beer descriptions to create the document-term matrix
    text_vectors = vectorizer.fit_transform(beers['beer_description'])

    # Select numerical columns
    numerical_columns = ['review_aroma', 'review_appearance', 'review_palate', 'review_taste']
    numerical_data = beers[numerical_columns]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the numerical data
    scaled_numerical_data = scaler.fit_transform(numerical_data)

    # Concatenate text and numerical vectors
    combined_vectors = hstack([text_vectors, scaled_numerical_data]).tocsr()

    return vectorizer, scaler, combined_vectors

# Load and preprocess data
beers = load_data()

# Parallelize preprocessing
def parallel_preprocess(beers):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(preprocess_data, beers)
        return future.result()

vectorizer, scaler, combined_vectors = parallel_preprocess(beers)

# Function to process data in chunks
def process_in_chunks(sparse_matrix, chunk_size=1000):
    num_rows = sparse_matrix.shape[0]
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = sparse_matrix[start:end]
        yield chunk

# Streamlit app
st.title("Find Similar Beers")

# Common beer styles with shortened names
common_styles = [
    "APA", "Pale Ale", "Amber Ale", "Brown Ale",
    "Porter", "Stout", "Dubbel", "Tripel",
    "Quadrupel", "Bitter", "English Porter", "Pilsner",
    "Hefeweizen", "Saison", "Imperial Stout"
]

# Initialize session state for selected beer style
if 'selected_style' not in st.session_state:
    st.session_state.selected_style = ""

# Display buttons for each beer style in the sidebar
st.sidebar.write("Select a Beer Style:")

# CSS to adjust button margins and spacing dynamically
button_style = """
    <style>
    .stButton button {
        white-space: nowrap;   /* Prevents text wrapping within buttons */
    }
    </style>
"""
st.sidebar.markdown(button_style, unsafe_allow_html=True)

# Dynamically create buttons row by row
buttons_per_row = 2  # Adjust as needed
rows = [common_styles[i:i + buttons_per_row] for i in range(0, len(common_styles), buttons_per_row)]

for row in rows:
    cols = st.sidebar.columns(len(row))  # Dynamically create the appropriate number of columns
    for col, style in zip(cols, row):
        with col:  # Place each button inside a column
            if st.button(style):  # Create button with text
                st.session_state.selected_style = style

# Checkbox to skip beer style input
skip_beer_style = st.checkbox("Skip Beer Style")

# Optional free-text input for beer style
if skip_beer_style:
    st.session_state.selected_style = ""
free_text_style = st.sidebar.text_input("Or enter a Beer Style:", st.session_state.selected_style)

# Use the free-text input if provided
if free_text_style:
    st.session_state.selected_style = free_text_style

# Display the selected beer style
if not skip_beer_style:
    st.sidebar.write(f"Selected Beer Style: {st.session_state.selected_style}")


# Take user input
user_aroma = st.sidebar.slider("Enter Aroma: ", min_value=0.0, max_value=5.0, step=0.1)
user_appearance = st.sidebar.slider("Enter Appearance: ", min_value=0.0, max_value=5.0, step=0.1)
user_palate = st.sidebar.slider("Enter Palate: ", min_value=0.0, max_value=5.0, step=0.1)
user_taste = st.sidebar.slider("Enter Taste: ", min_value=0.0, max_value=5.0, step=0.1)

if st.button("Find Similar Beers"):
    # Display a progress bar
    progress_bar = st.progress(0)
    
    # Combine user input text
    user_description = st.session_state.selected_style

    # Vectorize user input text
    user_text_vector = vectorizer.transform([user_description])

    # Scale user input numerical data
    user_numerical_data = scaler.transform([[user_aroma, user_appearance, user_palate, user_taste]])

    # Concatenate user text and numerical vectors
    user_vector = hstack([user_text_vector, user_numerical_data]).tocsr()

    # Compute cosine similarity in chunks
    similarities = []
    num_chunks = combined_vectors.shape[0] // 1000 + 1  # Calculate the number of chunks
    for i, chunk in enumerate(process_in_chunks(combined_vectors)):
        chunk_similarities = cosine_similarity(user_vector, chunk)
        similarities.append(chunk_similarities)
        progress_bar.progress((i + 1) / num_chunks)  # Update progress bar

    # Concatenate all similarity scores
    similarities = np.hstack(similarities)

    progress_bar.empty()  # Clear progress bar

    # Find the most similar beer
    most_similar_index = similarities.argmax()
    most_similar_beer = beers.iloc[most_similar_index]

    st.write("Most similar beer:")
    st.dataframe(most_similar_beer.to_frame().T)

    # Find the 10 most similar beers
    most_similar_indices = similarities[0].argsort()[-10:][::-1]
    most_similar_beers = beers.iloc[most_similar_indices]

    st.write("10 most similar beers:")
    st.dataframe(most_similar_beers)