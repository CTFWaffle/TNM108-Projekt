import sklearn as sk
import pandas as pd
import numpy as np


#----------------------Prepairing Data----------------------#
# Load the Beer_reviews dataset
beerdir = 'c:/Users/erika/Documents/GitHub/TNM108-Projekt/Datasets/beer_reviews.csv'

beers = pd.read_csv(beerdir)
      
#To lowercase
beers['beer_style'] = beers['beer_style'].str.lower()
#Sort beers by overall review, ascending
beers.sort_values(by='review_overall', ascending=False, inplace=True)
#Add a new column with the average review scoring
beers['avg_score'] = beers[['review_aroma', 'review_appearance', 'review_palate',  'review_taste']].mean(axis=1).round(1)
#Swap places for column beer_style and review_taste
beers = beers[['brewery_name','review_time', 'review_profilename','review_overall', 'review_aroma', 'review_appearance', 'review_palate',  'review_taste', 'avg_score','beer_style','beer_name', 'beer_abv']]
#Convert review_time to datetime
beers['review_time'] = pd.to_datetime(beers['review_time'], unit='s')

#Save the new dataset
beers.to_csv('c:/Users/erika/Documents/GitHub/TNM108-Projekt/Datasets/beer_reviews_cleaned.csv', index=False)