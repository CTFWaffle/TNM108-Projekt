import sklearn as sk
import pandas as pd
import numpy as np
import streamlit as st

#----------------------Prepairing Data----------------------#
# Load the dataset
beers = pd.read_csv('E:/Programmering/TNM108-Projekt/Datasets/beer_reviews_cleaned.csv')

# Beer categories without duplicates (104 categories)
my_beercategories = beers['beer_style'].unique()

#----------------------Main----------------------#
#Explanation:
#1. Combine Text Columns: Create a beer_description by combining relevant text columns.
#2. Vectorize Text Data: Use CountVectorizer to transform the text data into numerical vectors.
#3. Scale Numerical Data: Use StandardScaler to standardize the numerical columns.
#4. Concatenate Vectors: Use hstack from scipy.sparse to concatenate the text and numerical vectors.
#5. Process in Chunks: Define a function to process the sparse matrix in chunks. This function yields each chunk as a dense matrix.
#6. Process Each Chunk: Iterate over the chunks and process them as needed. For example, you can print the shape of each chunk or perform further analysis.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# Assuming 'beers' is a DataFrame and has the following columns:
# 'beer_style', 'beer_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste'

# Combine text columns to create a 'beer_description'
beers['beer_description'] = beers['beer_style']# + ' ' + beers['beer_name']

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

# Function to process data in chunks
def process_in_chunks(sparse_matrix, chunk_size=1000):
    num_rows = sparse_matrix.shape[0]
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = sparse_matrix[start:end].toarray()
        yield chunk

# Streamlit app
st.title("Find Similar Beers")

from sklearn.metrics.pairwise import cosine_similarity

# Take user input
user_beer_style = input("Enter Beer Style: ")
#user_beer_name = input("Enter Beer Name: ")
user_aroma = float(input("Enter Aroma (0-5): "))
user_appearance = float(input("Enter Appearance (0-5): "))
user_palate = float(input("Enter Palate (0-5): "))
user_taste = float(input("Enter Taste (0-5): "))

if st.button("Find Similar Beers"):
    # Combine user input text
    user_description = user_beer_style #+ ' ' + user_beer_name

    # Vectorize user input text
    user_text_vector = vectorizer.transform([user_description])

    # Scale user input numerical data
    user_numerical_data = scaler.transform([[user_aroma, user_appearance, user_palate, user_taste]])

    # Concatenate user text and numerical vectors
    user_vector = hstack([user_text_vector, user_numerical_data]).toarray()

    # Compute cosine similarity in chunks
    similarities = []
    for chunk in process_in_chunks(combined_vectors):
        chunk_similarities = cosine_similarity(user_vector, chunk)
        similarities.append(chunk_similarities)

    # Concatenate all similarity scores
    similarities = np.hstack(similarities)

    # Find the 5 most similar beers
    most_similar_indices = similarities[0].argsort()[-5:][::-1]
    most_similar_beers = beers.iloc[most_similar_indices]

    st.write("5 most similar beers:")
    st.write(most_similar_beers)

    # Find the most similar beer
    most_similar_index = similarities.argmax()
    most_similar_beer = beers.iloc[most_similar_index]

    st.write("Most similar beer:")
    st.write(most_similar_beer)