import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import dump

# Load MovieLens dataset (you can also use your own dataset)
data = Dataset.load_builtin('ml-100k')

# Define a reader for the dataset
reader = Reader(rating_scale=(1, 5))

# Load the data into a DataFrame
df = pd.DataFrame(data.raw_ratings, columns=['user_id', 'item_id', 'rating', 'timestamp'])

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build the SVD model
model = SVD()

# Train the model
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)

# Save the model
dump.dump('movie_recommender', algo=model)

# Function to get movie recommendations
def get_movie_recommendations(user_id, n_recommendations=5):
    if user_id not in df['user_id'].unique():
        return "User ID not found in the dataset."
    
    user_movies = df[df['user_id'] == user_id]
    
    if user_movies.empty:
        return "This user has not rated any movies."
    
    user_movie_ids = user_movies['item_id'].unique()
    all_movie_ids = df['item_id'].unique()
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in user_movie_ids]
    
    recommendations = []
    for movie_id in unrated_movie_ids:
        pred_rating = model.predict(user_id, movie_id).est
        recommendations.append((movie_id, pred_rating))
    
    # Sort by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations[:n_recommendations]

# Main execution
if __name__ == "__main__":
    try:
        user_id = int(input("Enter your User ID: "))  # Prompt user for their User ID
        recommended_movies = get_movie_recommendations(user_id)
        print("Recommended Movies for User ID {}: {}".format(user_id, recommended_movies))
    except ValueError:
        print("Please enter a valid numeric User ID.")
