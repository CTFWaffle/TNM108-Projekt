import sklearn as sk
import pandas as pd
import numpy as np

#----------------------Prepairing Data----------------------#
# Load the dataset
beers = pd.read_csv('E:/Programmering/TNM108-Projekt/Datasets/beer_reviews_cleaned.csv')

# Beer categories without duplicates (104 categories)
my_beercategories = beers['beer_style'].unique()

#----------------------Functions----------------------#
#Demand user input Aroma, appearance, palate, and taste
#Return the beer with the closest scores
#Find the beer with the closest score for each category
def collect_userData(aroma, appearance, palate, taste, style):
    # Find the beer with the closest score for each category
    '''
    aroma_beer = beers.iloc[(beers['review_aroma'] - aroma).abs().argsort()[:1]]
    appearance_beer = beers.iloc[(beers['review_appearance'] - appearance).abs().argsort()[:1]]
    palate_beer = beers.iloc[(beers['review_palate'] - palate).abs().argsort()[:1]]
    taste_beer = beers.iloc[(beers['review_taste'] - taste).abs().argsort()[:1]]
    '''

    #print("User input: " + style)
    found_style = "No styles found"
    found_stylez = []
    # Find the beer style where user input is at all featured
    for i in range(len(my_beercategories)):
        #If style is found within a string add it to found_style
        if(str.find(my_beercategories[i], style)!=-1):
            #print("Found style: " + my_beercategories[i])
            found_stylez.append(my_beercategories[i]) 

    if(len(found_stylez)>0):
        found_style = found_stylez
    # Return the beer with the closest score for each category
    return aroma, appearance, palate, taste, found_style

# Print the beer style of a unique beer
# If the beer has an index, print the beer style
# If the beer has no index, print nothing
def print_beer(index):
    for i in range(len(beers)):
        if(i==index):
            #Print out everything in the row with index i
            #Rename column names
            beers.columns = ['Brewery Name', 'Review Time', 'Review Platform', 'Overall Review', 'Aroma', 'Appearance', 'Palate', 'Taste', 'Average Review Score', 'Beer Style', 'Beer Name', 'Alcohol by Volume']
            print(beers.loc[i])
    


#----------------------Main----------------------#
#print(beers.head().to_string(index=False, justify='center') + "\n")

#print(beers.columns) # Print the column names (categories)

#print(beers.head().to_string(index=False, justify='center') + "\n")
#print(my_beercategories) # All beer categories with duplicates

#print(my_beercategories)
#print(len(my_beercategories))

#print_beer(248550)
'''
arom,appearance,palate,taste,style = collect_userData(np.float64(input("Enter Aroma (0-5): ")), np.float64(input("Enter Appearance (0-5): ")), np.float64(input("Enter Palate (0-5): ")), np.float64(input("Enter Taste (0-5): ")), str.lower(input("Enter Beer Style: ")))

print('All Beer styles found:',style) #prints all styles with the related term (ex Belgian)


collectedBeer = [arom, appearance, palate, taste]


#Standardize the output
for i in range(len(collectedBeers)):
    print(collectedBeers[i].to_string(index=False,  justify='center' ,col_space=10) + "\n")

'''
#Retrieve closest beer of each style

#----------------------TF-IDF----------------------#

#print multiple columns of the first beer in the dataset
#print(beers[['review_aroma', 'review_appearance', 'review_palate', 'review_taste','beer_style']].iloc[0])
#print(beers.iloc[0])
#input("Press Enter to continue...")

#----------------------CoPilot----------------------#
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
beers['beer_description'] = beers['beer_style'] #+ ' ' + beers['beer_name']

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
        chunk = sparse_matrix[start:end].todense()
        yield chunk

from sklearn.metrics.pairwise import cosine_similarity

# Take user input
user_beer_style = input("Enter Beer Style: ")
#user_beer_name = input("Enter Beer Name: ")
user_aroma = float(input("Enter Aroma (0-5): "))
user_appearance = float(input("Enter Appearance (0-5): "))
user_palate = float(input("Enter Palate (0-5): "))
user_taste = float(input("Enter Taste (0-5): "))

# Combine user input text
user_description = user_beer_style #+ ' ' + user_beer_name

# Vectorize user input text
user_text_vector = vectorizer.transform([user_description])

# Scale user input numerical data
user_numerical_data = scaler.transform([[user_aroma, user_appearance, user_palate, user_taste]])

# Concatenate user text and numerical vectors
user_vector = hstack([user_text_vector, user_numerical_data]).tocsr()

# Compute cosine similarity between user vector and combined vectors
similarities = cosine_similarity(user_vector, combined_vectors)

# Find the 5 most similar beers
most_similar_indices = similarities[0].argsort()[-5:][::-1]
most_similar_beers = beers.iloc[most_similar_indices]

print("5 most similar beers:")
print(most_similar_beers)

# Find the most similar beer
most_similar_index = similarities.argmax()
most_similar_beer = beers.iloc[most_similar_index]

print("Most similar beer:")
print(most_similar_beer)


#-----------------------------------------------From lab 4------------------------------------------------#
'''
tfidf_vectorizer = TfidfVectorizer()


d1 = beers.iloc[0]
d2 = beers.iloc[1]
d3 = beers.iloc[2]
d4 = beers.iloc[3]

Z = (d1, d2, d3, d4)
print(d1)
#print(Z)
input("Press Enter to continue...") #Pause
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

input("Press Enter to continue...") #Pause

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(Z[1], Z)
print(cos_similarity)
input("Press Enter to continue...") #Pause

#Z is now filled with vectors of the first 4 beers in the dataset
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
#print(vectorizer)

my_stop_words={"the", "is"}
my_vocabulary={'blue':0, 'sun':1, 'bright':2, 'sky':3}
vectorizer = CountVectorizer(stop_words=my_stop_words, vocabulary=my_vocabulary)

print(vectorizer.vocabulary)
print(vectorizer.stop_words)

smatrix = vectorizer.transform(Z)
print(smatrix)

matrix = smatrix.todense()
print(matrix)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# Print idf values
feature_names = vectorizer.get_feature_names_out()
import pandas as pd
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=feature_names, columns=["idf_weights"])

# Sort ascending
df_idf = df_idf.sort_values(by=["idf_weights"])
print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document_vector = tf_idf_vector[0] # first document "The sky is blue."
# print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df = df.sort_values(by=["tfidf"], ascending=False)
print(df)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)
'''