import os
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PowerTransformer, LabelEncoder

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

# Define the output paths
output_path = "../output/dataPreprocessing"
os.makedirs(output_path, exist_ok=True)

def preprocess_data(data_path):
    """
    Preprocess training and testing data from files in the specified path.

    Args:
        data_path (str): Path containing the train and test data files.

    Returns:
        tuple: Preprocessed training and testing datasets.
    """

    random_state = 42

    # Redirect console outputs to a text file
    log_file = os.path.join(output_path, 'output.txt')
    with open(log_file, 'w') as log:
        def log_print(*args):
            print(*args)
            print(*args, file=log)
        
        # Load the train dataset
        data_file_train = "fraudTrain.csv"
        data_path_train = os.path.join(data_path, data_file_train)
        log_print(f"Loading dataset from {data_path_train}...")
        train_df = pd.read_csv(data_path_train)

        # Display the first few rows of the dataset
        log_print(f"First few rows of the dataset:")
        log_print(train_df.head())

        # Load the test dataset
        data_file_train_test = "fraudTest.csv"
        data_path_test = os.path.join(data_path, data_file_train_test)
        log_print(f"Loading dataset from {data_path_test}...")
        test_df = pd.read_csv(data_path_test)

        # Display the first few rows of the dataset
        log_print(f"First few rows of the dataset:")
        log_print(test_df.head())

        # Define the target column (dependent variable)
        target_column = "is_fraud"

        # Print the shapes of the training and testing data to confirm the separation of data into train and test sets
        log_print(f"Train data shape: {train_df.shape}")
        log_print(f"Test data shape: {test_df.shape}")

        # Select feature columns (independent variables) from the training data to create the training set (X_train)
        X_train_raw = train_df.drop(columns=target_column, axis=1)

        # Select target columns (dependent variables) from the training data to create the target set (y_train)
        y_train = train_df[target_column]

        # Select feature columns (independent variables) from the test data to create the test set (X_test)
        X_test_raw = test_df.drop(columns=target_column, axis=1)

        # Select target columns (dependent variables) from the test data to create the target set (y_test)
        y_test = test_df[target_column]

        # Print the shapes of the training and testing datasets to verify the number of rows and columns for each
        log_print(f"Shape of X_train_raw: {X_train_raw.shape}")
        log_print(f"Shape of X_test_raw: {X_test_raw.shape}")
        log_print(f"Shape of y_train: {y_train.shape}")
        log_print(f"Shape of y_test: {y_test.shape}")

        class ChangeDataType(BaseEstimator, TransformerMixin):
            # Converts specified columns to datetime format.
            def __init__(self, columns):
                self.columns = columns

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X = X.copy()
                for col in self.columns:
                    X[col] = pd.to_datetime(X[col], errors='coerce')
                return X

        class DateTimeFeatures(BaseEstimator, TransformerMixin):
            # Extracts date and time-related features like hour, month, day of the week, and part of the day.
            def __init__(self, date_column, transaction_hour_bins, transaction_hour_labels):
                self.date_column = date_column
                self.transaction_hour_bins = transaction_hour_bins
                self.transaction_hour_labels = transaction_hour_labels
                self.new_columns = ['transaction_hour', 'transaction_month', 'is_weekend', 'day_of_week', 'part_of_day']

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X = X.copy()
                X['transaction_hour'] = X[self.date_column].dt.hour
                X['transaction_month'] = X[self.date_column].dt.month
                X['is_weekend'] = X[self.date_column].dt.weekday.isin([5, 6]).astype(int)

                # Day of week: Monday=0, Sunday=6
                X['day_of_week'] = X[self.date_column].dt.day_name()

                # Part of day classification
                X['part_of_day'] = pd.cut(X['transaction_hour'],
                                        bins=self.transaction_hour_bins,
                                        labels=self.transaction_hour_labels,
                                        right=True)
                return X

        class AgeFeature(BaseEstimator, TransformerMixin):
            # Calculates age based on the date of birth (DOB) column.
            def __init__(self, dob_column):
                self.dob_column = dob_column
                self.new_column = 'age'

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X = X.copy()
                reference_date = pd.Timestamp(2020, 12, 31)
                X[self.new_column] = (reference_date - X[self.dob_column]).dt.days // 365
                return X

        class CalculateDistance(BaseEstimator, TransformerMixin):
            # Calculates the distance between two geographical points using the Haversine formula.
            def __init__(self, lat_col, long_col, merch_lat_col, merch_long_col):
                self.lat_col = lat_col
                self.long_col = long_col
                self.merch_lat_col = merch_lat_col
                self.merch_long_col = merch_long_col
                self.new_column = 'distance'

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X = X.copy()

                # Convert latitudes and longitudes to radians
                lat1 = np.radians(X[self.lat_col])
                lon1 = np.radians(X[self.long_col])
                lat2 = np.radians(X[self.merch_lat_col])
                lon2 = np.radians(X[self.merch_long_col])

                # Haversine formula to calculate distance
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                R = 6371  # Radius of the Earth in kilometers
                X[self.new_column] = R * c  # Distance in kilometers

                return X

        class BinCityPopulation(BaseEstimator, TransformerMixin):
            # Groups city population into bins with specified labels.
            def __init__(self, city_pop_bins, city_pop_labels):
                self.city_pop_bins = city_pop_bins
                self.city_pop_labels = city_pop_labels
                self.new_column = 'city_pop_bin'

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X = X.copy()
                X[self.new_column] = pd.cut(X['city_pop'], bins=self.city_pop_bins, labels=self.city_pop_labels)
                return X

        class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
            # Applies the Yeo-Johnson transformation to normalize the 'amt' column.
            def __init__(self):
                self.transformer = PowerTransformer(method='yeo-johnson')
                self.new_column = 'amt_yeo_johnson'

            def fit(self, X, y=None):
                self.transformer.fit(X[['amt']])
                return self

            def transform(self, X):
                X = X.copy()
                X[self.new_column] = self.transformer.transform(X[['amt']])
                return X


        class DropColumns(BaseEstimator, TransformerMixin):
            # Drops specified columns from the dataset.
            def __init__(self, columns):
                self.columns = columns

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X = X.drop(columns=self.columns, errors='ignore')
                self.remaining_columns = X.columns
                return X

        class LabelEncoding(BaseEstimator, TransformerMixin):
            # Performs label encoding for specified categorical columns.
            def __init__(self, columns):
                self.columns = columns
                self.label_encoders = {}

            def fit(self, X, y=None):
                for col in self.columns:
                    le = LabelEncoder()
                    le.fit(X[col])
                    self.label_encoders[col] = le
                return self

            def transform(self, X):
                X = X.copy()
                for col in self.columns:
                    X[col] = self.label_encoders[col].transform(X[col])
                return X


        class ScaleFeatures(BaseEstimator, TransformerMixin):
            # Scales numerical features to a range of 0 to 1 using MinMaxScaler.
            def __init__(self):
                self.scaler = MinMaxScaler()

            def fit(self, X, y=None):
                self.scaler.fit(X)
                return self

            def transform(self, X):
                X = X.copy()
                X[:] = self.scaler.transform(X)
                return X

        # Preprocessing pipeline
        city_pop_bins = [0, 10000, 50000, 100000, 500000, 1000000, np.inf]
        city_pop_labels = ['<10K', '10K-50K', '50K-100K', '100K-500K', '500K-1M', '>1M']

        transaction_hour_bins=[-1, 5, 11, 17, 21, 24]
        transaction_hour_labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']

        drop_columns = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'amt',
                        'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long',
                        'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']

        categorical_features = ['category', 'gender', 'day_of_week', 'part_of_day', 'city_pop_bin']

        # Preprocessing pipeline
        preprocessor = Pipeline([
            ('change_dtype', ChangeDataType(columns=['trans_date_trans_time', 'dob'])),
            ('datetime_features', DateTimeFeatures(date_column='trans_date_trans_time',
                                                transaction_hour_bins=transaction_hour_bins,
                                                transaction_hour_labels=transaction_hour_labels)),
            ('age_feature', AgeFeature(dob_column='dob')),
            ('calculate_distance', CalculateDistance(lat_col='lat', long_col='long',
                                                    merch_lat_col='merch_lat', merch_long_col='merch_long')),
            ('bin_city_pop', BinCityPopulation(city_pop_bins=city_pop_bins, city_pop_labels=city_pop_labels)),
            ('yeo_johnson', YeoJohnsonTransformer()),
            ('drop_columns', DropColumns(columns=drop_columns)),
            ('label_encoding', LabelEncoding(columns=categorical_features)),
            ('scale_features', ScaleFeatures()),
        ])

        # Fit and transform the training data
        train_preprocessed = preprocessor.fit_transform(X_train_raw)
        train_preprocessed[target_column] = y_train.values
        log_print("Training data preprocessing completed.")

        # Save the preprocessed training data
        train_preprocessed_path = os.path.join(data_path, "train_preprocessed.pkl")
        with open(train_preprocessed_path, 'wb') as pkl_file:
            pickle.dump(train_preprocessed, pkl_file)
        log_print(f"Preprocessed training data saved to {train_preprocessed_path}")

        # Transform the test data
        test_preprocessed = preprocessor.transform(X_test_raw)
        test_preprocessed[target_column] = y_test.values
        log_print("Test data preprocessing completed.")

        # Save the preprocessed test data
        test_preprocessed_path = os.path.join(data_path, "test_preprocessed.pkl")
        with open(test_preprocessed_path, 'wb') as pkl_file:
            pickle.dump(test_preprocessed, pkl_file)
        log_print(f"Preprocessed test data saved to {test_preprocessed_path}")

    return train_preprocessed, test_preprocessed

def apply_resampling(data_path, train_preprocessed):
    """
    Apply SMOTE and ADASYN resampling to the training data.

    Args:
        train_preprocessed (pd.DataFrame): Preprocessed training dataset.

    Returns:
        tuple: SMOTE and ADASYN resampled datasets.
    """
    log_file = os.path.join(output_path, 'output.txt')
    target_column = "is_fraud"
    random_state = 42  # Set a random state for reproducibility
    with open(log_file, 'a') as log:
        def log_print(*args):
            print(*args)
            print(*args, file=log)
        # Class to oversample the minority class using the Synthetic Minority Over-sampling Technique (SMOTE)
        class SMOTESampler:
            def __init__(self, target_column):
                self.target_column = target_column
                self.sampler = SMOTE(random_state=random_state)

            # Fits the sampler and resamples the data to balance the target column
            def fit_resample(self, df):
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
                return X_resampled.assign(**{self.target_column: y_resampled})

        # Class to oversample the minority class using the Adaptive Synthetic (ADASYN) method
        class ADASYN_Sampler:
            def __init__(self, target_column):
                self.target_column = target_column
                self.sampler = ADASYN(random_state=random_state)

            # Fits the sampler and resamples the data to balance the target column
            def fit_resample(self, df):
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
                return X_resampled.assign(**{self.target_column: y_resampled})

        # Class to reduce data imbalance by removing Tomek Links (overlapping majority samples near minority samples)
        class TomekLinksSampler:
            def __init__(self, target_column):
                self.target_column = target_column
                self.sampler = TomekLinks()

            # Fits the sampler and resamples the data to reduce Tomek Links
            def fit_resample(self, df):
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
                return X_resampled.assign(**{self.target_column: y_resampled})

        # Class to combine SMOTE oversampling and Tomek Links removal for handling imbalanced data
        class SMOTETomekSampler:
            def __init__(self, target_column):
                self.target_column = target_column
                self.sampler = SMOTETomek(random_state=random_state)

            # Fits the sampler and resamples the data using a combination of SMOTE and Tomek Links
            def fit_resample(self, df):
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]
                X_resampled, y_resampled = self.sampler.fit_resample(X, y)
                return X_resampled.assign(**{self.target_column: y_resampled})

        # Apply SMOTE
        smote_sampler = SMOTESampler(target_column=target_column)
        smote_resampled_df = smote_sampler.fit_resample(train_preprocessed)
        log_print("SMOTE completed")

        # Apply ADASYN
        adasyn_sampler = ADASYN_Sampler(target_column=target_column)
        adasyn_resampled_df = adasyn_sampler.fit_resample(train_preprocessed)
        log_print("ADASYN completed")

        # Save datasets as pickle files
        datasets = {
            "SMOTE": smote_resampled_df,
            "ADASYN": adasyn_resampled_df,
        }

        for name, dataset in datasets.items():
            file_path = os.path.join(data_path, f"{name}_resampled.pkl")
            with open(file_path, 'wb') as pkl_file:
                pickle.dump(dataset, pkl_file)
            log_print(f"{name} dataset saved to {file_path}")

        # Inspect the resampled datasets
        log_print(smote_resampled_df[target_column].value_counts())
        log_print(adasyn_resampled_df[target_column].value_counts())

        log_print("Data preprocessing completed. Datasets are saved for model training.")

    return smote_resampled_df, adasyn_resampled_df
