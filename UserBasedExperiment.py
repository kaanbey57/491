import pandas as pd
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate
import time
import os

#For this experiment while writing the code, I used generative AI to support my code. In other words, if something was wrong or I did not know, I asked it to AI.
FILE_PATH = 'sales_CLEAN_final.csv'
IMPLICIT_RATING = 1.0
K_NEIGHBORS = 40 # Number of neighbors to consider
CV_FOLDS = 5 # Number of folds for cross-validation

def load_data():

        df = pd.read_csv(FILE_PATH)
        data_for_cf = df[['TklMasterId', 'HotelId']].copy()      #we get the data from our csv file
        data_for_cf.rename(columns={'TklMasterId': 'user_id', 'HotelId': 'item_id'}, inplace=True)
        
        data_for_cf['rating'] = IMPLICIT_RATING  #here we add implicit rating
        
        data_for_cf.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)  #tried to filter duplicates

        return data_for_cf


def run_ubcf_experiment():

    df_data = load_data()

    reader = Reader(rating_scale=(IMPLICIT_RATING, IMPLICIT_RATING))
    data = Dataset.load_from_df(df_data[['user_id', 'item_id', 'rating']], reader)

    user_based_sim_options = {
        'name': 'cosine',      # Use Cosine Similarity
        'user_based': True,    # *** CRITICAL: Enables User-Based Nearest Neighbor ***
        'min_support': 1       # Minimum number of common items needed to compute similarity
    }

    algo = KNNBasic(sim_options=user_based_sim_options, k=K_NEIGHBORS)

    # 3. Execution and Evaluation
    print(f"Running User-Based Collaborative Filtering (KNN-User) with {CV_FOLDS}-fold cross-validation...")
    print(f"Total Unique Interactions: {len(df_data)}")
    print(f"Unique Users: {df_data['user_id'].nunique()}")
    print(f"Unique Hotels: {df_data['item_id'].nunique()}")

    start_time = time.time()
    
    cv_results = cross_validate(
        algo,
        data,
        measures=['RMSE'],
        cv=CV_FOLDS,                 
        verbose=True  # Display verbose output to see fold times
    )
    
    end_time = time.time()
    total_run_time = end_time - start_time

    rmse_values = cv_results['test_rmse']
    fit_times = cv_results['fit_time']
    
    average_rmse = sum(rmse_values) / len(rmse_values)
    average_fit_time = sum(fit_times) / len(fit_times)

    print("\n--- Final Results Summary ---")
    print(f"Algorithm: User-Based KNN (k={K_NEIGHBORS}, Cosine)")
    print(f"Average RMSE (Expected ≈ {IMPLICIT_RATING:.4f} or 0.0000): {average_rmse:.4f}")
    print(f"Average Fit Time per fold (Complexity Metric): {average_fit_time:.4f}s")
    print(f"Total Run Time for {CV_FOLDS}-fold CV: {total_run_time:.4f}s")
    print("\nConclusion: The Average Fit Time is the critical result to compare against the 0.35s Model-Based (SVD) time.")

if __name__ == '__main__':
    run_ubcf_experiment()

