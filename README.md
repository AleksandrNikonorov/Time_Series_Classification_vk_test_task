# Time Series Classification

## Project Description

1. **Main Idea** – Take the longest time span of the time series, divide it into 8 intervals (by years), and compute statistics for the time series defined on each interval. If a time interval has no value for the series, it is filled with NaN. As a result, a dataframe with 96 features (8 intervals with 12 statistics each) is produced. Since this dataframe cannot be uploaded to GitHub and the creation process takes some time, the code along with links to download the preprocessed `train.parquet` and `test.parquet` files are provided in the **`load_dataframe.py`** file.

2. The CatBoost classifier has proven to perform best. In **`my_model_selection.py`**, a class is implemented to select the best training components (dataset, model, and hyperparameters). The result is saved in a table called `result`.

3. **`MainNavigation`** – This is the main module of the project that combines all functionalities:
   - ``create_datasets(train_path, test_path)``: Initiates the process of creating the final datasets.
   - ``run_model_selection``: Executes a grid search over datasets, models, and hyperparameters, trains the best model on the entire dataset, and saves it.
   - ``get_submission``: Uses the trained model to make predictions on the test data and outputs them to the file `submission.csv`.

## Project Structure

1. **`load_dataframe`**:
   - A function for loading the preprocessed datasets from Google Drive.

2. **`MainNavigation`**:
   - The main module for running the models.

3. **`create_dataset_functions`**:
   - A collection of functions for generating features based on the time series.

4. **`my_model_selection`**:
   - A class for selecting models and comparing results across different datasets, models, and their hyperparameters.

5. **`EDA`**:
   - A module for exploratory data analysis of the time series. It helps investigate the data structure before using it in the model. (This part of the work has been started but is not yet complete.)

6. **`trained_BestModel_catboost_on_rolling_Intervals_12Stats_0.pkl`**:
   - A pre-trained model.

## Data Preparation

1. **Download Data**:  
   The datasets are preprocessed and stored on Google Drive. To download them, use the `load_dataframe` function, which provides access to the necessary data.

2. **Predictions**:  
   Run the script from MainNavigation to create the `submission.csv` file.


