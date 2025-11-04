# MLB Prediction using Deep Learning Models
## Dataset (/data)
* **game_logs_2024.csv** : This is the data of all the MLB games played during the year 2024, retrieved from statsapi coded in load_data.ipynb.
* **pitching_stats_2024.csv** : This is the pitching data of the MLB pitchers, retrieved from https://www.baseball-reference.com/leagues/majors/2024-standard-pitching.shtml#players_standard_pitching
* **team_batting_stats_2024.csv** : This is the batting data of the MLB teams, retrieved from https://www.baseball-reference.com/leagues/majors/2024-standard-batting.shtml
* **final_game_logs.csv** : Performed Exploratory Data Analysis(EDA) in eda.ipynb, picked relevant features that contribute to game results, and processed data from 'preprocessing.ipynb' to use for training.
  
## Jupyter Notebook Files (/notebooks)
* **load_data.ipynb** : The notebook for accessing statsapi, for retrieving game results.
* **eda.ipynb** : Performed Exploratory Data Analysis (EDA) and chose the relevant features for batting/pitching that contributes to wins/losses.
* **preprocessing.ipynb** : Cleaned up missing/amiguous data, merged the game results with its corresponding pitching/batting data, stored the final dataset as 'final_game_logs.csv'.
* **weather.ipynb** : Scraped off weather data from https://www.baseball-reference.com/
* **weather_onehot.ipynb** : One-hot encoding the categorical weather columns
* **preprocessing_weather.ipynb** : Preprocessing weather data and augmenting to the final dataset
* **last_games.ipynb** : Augemented the last 7 games win percentage for each game, and the win/lose streak as well
* **lineup_score.ipynb** : Scraped off the starting lineup for each game, computed the mean OPS for each lineup


## Python Files (/src)
* **preprocessing.py** : Loads the processed dataset 'final_game_logs.csv', cleans missing data and retrieves numerical columns for training, scales using StandardScaler().
* **model.py**
   * build_model() : Builds an Artificial Neural Network (ANN) using Tensorflow/Keras.
   * visualize() : Visualizes the training history and the model's performance.
* **evaluate.py** : Evaluating models in (../saved_models) with 2024 data
* **evaluate_2025.py** : Evaluating models in (../saved_models) with 2025 data
* **grid_search_epochs** : Finding the proper epochs for each models

## Evaluation Reports
