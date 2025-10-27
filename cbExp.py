import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity

inputFile = "/home/kaan/Desktop/StayWise/ContentBasedExperiment/hotel_features.csv"
similarHotelID = [522, 3107]   #hotels that user liked
topN = 10  #top n hotels to show

hotelFeatures = pd.read_csv(inputFile, sep=",", index_col = "HotelId")

likedFeatures = hotelFeatures.loc[similarHotelID]

userProfileVector = likedFeatures.mean(axis=0)  # we do axis=0 to calculate the mean of each column, if it is axis=1, then it will calculate the mean of each row

userProfile2D = userProfileVector.values.reshape(1, -1) #make 1D array to 2D array because cos sim expects 2D array (1 row with -1 (as many as) columns)

similarityScore = cosine_similarity(userProfile2D, hotelFeatures)
similarityScore = similarityScore.flatten()  # we have to flatten it because cos sim return [[x,y,z]] so we have to turn [x,y,z]

hotelSimilarities = pd.Series(similarityScore, index = hotelFeatures.index) #create a excel like thing with each hotel index corresponds to some simScore
hotelSimilarities = hotelSimilarities.sort_values(ascending = False)
recommendations = hotelSimilarities[~hotelSimilarities.index.isin(similarHotelID)] # exclude the user's already liked hotels

print(recommendations.head(topN))
print(hotelFeatures.loc[similarHotelID])