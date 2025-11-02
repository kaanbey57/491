import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import time

sales_file = 'sales_CLEAN_final.csv' 

#Load Input Data
df_sales = pd.read_csv(sales_file)

#Here we prepare the data for surprise
df_implicit = df_sales[['TklMasterId', 'HotelId']].copy()
df_implicit['rating'] = 1.0  # Unfortunately, since our dataset does not contain ratings, we chose 1 if somebody bought something

reader = Reader(rating_scale=(1, 1))

data = Dataset.load_from_df(df_implicit, reader)

#I chose SVD algorithm
algo = SVD() 

# Run the experiment
results = cross_validate(algo, data, measures=['rmse'], cv=5, verbose=True)

# Measure results
avg_rmse = np.mean(results['test_rmse'])
avg_fit_time = np.mean(results['fit_time'])
avg_test_time = np.mean(results['test_time'])

print(f"\nAverage RMSE (Root Mean Squared Error): {avg_rmse:.4f}")
print(f"Average Fit Time (Training Time): {avg_fit_time:.4f} seconds")
print(f"Average Test Time: {avg_test_time:.4f} seconds")

