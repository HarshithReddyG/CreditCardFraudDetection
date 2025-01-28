import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
import math

def perform_eda(df, output_path):
    """
    Perform exploratory data analysis (EDA) on the input DataFrame and save outputs.

    Args:
        df (pd.DataFrame): Input DataFrame.
        output_path (str): Path to save EDA outputs (logs and visualizations).

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, 'eda_output.txt')
    with open(log_file, 'w') as log:
        def log_print(*args):
            print(*args)
            print(*args, file=log)

        log_print("Starting Exploratory Data Analysis (EDA)...")

        df_eda = df.copy()

        # Convert 'trans_date_trans_time' to datetime format
        df_eda['trans_date_trans_time'] = pd.to_datetime(df_eda['trans_date_trans_time'])

        # Hour of the Transaction
        df_eda['transaction_hour'] = df_eda['trans_date_trans_time'].dt.hour

        # Day of the Week
        df_eda['day_of_week'] = df_eda['trans_date_trans_time'].dt.day_name()  # Keep only weekday names

        # Month of the Transaction
        df_eda['transaction_month'] = df_eda['trans_date_trans_time'].dt.month

        # Part of the Day
        def categorize_part_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            elif 21 <= hour or hour < 5:
                return 'Night'

        df_eda['part_of_day'] = df_eda['transaction_hour'].apply(categorize_part_of_day)

        # Is Weekend
        df_eda['is_weekend'] = df_eda['trans_date_trans_time'].dt.weekday.isin([5, 6]).astype(int)

        # Convert 'dob' to datetime
        df_eda['dob'] = pd.to_datetime(df_eda['dob'], errors='coerce')

        # Calculate age as of January 20, 2025
        reference_date = datetime(2025, 1, 20)  # Set the reference date to 20th January 2025
        df_eda['age'] = (reference_date - df_eda['dob']).dt.days // 365  # Convert days to years

        # Create a more detailed age group column
        def categorize_age_group(age):
            if age <= 12:
                return 'Child'
            elif 13 <= age <= 17:
                return 'Teenager'
            elif 18 <= age <= 25:
                return 'Young Adult'
            elif 26 <= age <= 40:
                return 'Adult'
            elif 41 <= age <= 60:
                return 'Middle-Aged'
            elif 61 <= age <= 80:
                return 'Senior'
            else:
                return 'Elderly'

        df_eda['age_group'] = df_eda['age'].apply(categorize_age_group)

        # Function to calculate distance between cardholder and merchant
        def calculate_distance(X, lat_col, long_col, merch_lat_col, merch_long_col, new_column):
            """
            Calculate the distance (in kilometers) between two geographic coordinates using the Haversine formula.
            """
            X = X.copy()

            # Convert latitudes and longitudes to radians
            lat1 = np.radians(X[lat_col])
            lon1 = np.radians(X[long_col])
            lat2 = np.radians(X[merch_lat_col])
            lon2 = np.radians(X[merch_long_col])

            # Haversine formula to calculate distance
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            R = 6371  # Radius of the Earth in kilometers
            X[new_column] = R * c  # Distance in kilometers

            return X 
        
        # Apply the function to calculate distance for each transaction
        df_eda = calculate_distance(df_eda, 'lat', 'long', 'merch_lat', 'merch_long', 'distance')

        # Create bins for 'city_pop'
        city_pop_bins = [0, 10000, 50000, 100000, 500000, 1000000, np.inf]
        city_pop_labels = ['<10K', '10K-50K', '50K-100K', '100K-500K', '500K-1M', '>1M']

        # Bin the 'city_pop' column
        df_eda['city_pop_bin'] = pd.cut(df_eda['city_pop'], bins=city_pop_bins, labels=city_pop_labels)

        # Plot the distribution of the target variable
        plt.figure(figsize=(8, 6))
        sns.countplot(x='is_fraud', data=df_eda, palette='viridis')
        plt.title('Distribution of Fraudulent Transactions')
        plt.xlabel('Fraudulent Transaction')
        plt.ylabel('Count')
        fraud_dist_path = os.path.join(output_path, 'fraud_distribution.png')
        plt.savefig(fraud_dist_path)
        plt.close()
        log_print(f"Fraudulent transaction distribution plot saved to {fraud_dist_path}")

        # Plot the distribution of the transaction amount
        plt.figure(figsize=(8, 6))
        sns.histplot(df_eda['amt'], kde=True, color='red', bins=50)
        plt.title('Distribution of Transaction Amounts')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        amt_dist_path = os.path.join(output_path, 'transaction_amount_distribution.png')
        plt.savefig(amt_dist_path)
        plt.close()
        log_print(f"Transaction amount distribution plot saved to {amt_dist_path}")

        # Group by 'is_fraud' and calculate the mean transaction amount
        avg_amt_fraud = df_eda.groupby('is_fraud')['amt'].mean()

        # Plot the result
        avg_amt_fraud_path = os.path.join(output_path, 'avg_transaction_amount_by_fraud_status.png')
        avg_amt_fraud.plot(kind='bar', color=['green', 'red'], figsize=(6, 4))
        plt.title('Average Transaction Amount by Fraud Status')
        plt.xlabel('Fraud Status')
        plt.ylabel('Average Amount')
        plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'], rotation=0)
        plt.savefig(avg_amt_fraud_path)
        plt.close()
        log_print(f"Average transaction amount by fraud status plot saved to {avg_amt_fraud_path}")

        # Group by 'day_of_week' and calculate the sum of fraudulent transactions
        fraud_by_day = df_eda.groupby('day_of_week')['is_fraud'].sum()

        # Plot the result
        plt.figure(figsize=(8, 6))
        fraud_by_day.sort_values().plot(kind='bar', color='yellow')
        plt.title('Fraudulent Transactions by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Fraudulent Transactions')
        fraud_by_day_path = os.path.join(output_path, 'fraud_by_day_of_week.png')
        plt.savefig(fraud_by_day_path)
        plt.close()
        log_print(f"Fraudulent transactions by day of week plot saved to {fraud_by_day_path}")

        # Group by age and calculate the sum of fraudulent transactions
        fraud_by_age = df_eda.groupby('age')['is_fraud'].sum()
        plt.figure(figsize=(8, 6))
        fraud_by_age.plot(kind='line', color='red')
        plt.title('Fraudulent Transactions by Age')
        plt.xlabel('Age')
        plt.ylabel('Number of Fraudulent Transactions')
        fraud_by_age_path = os.path.join(output_path, 'fraud_by_age.png')
        plt.savefig(fraud_by_age_path)
        plt.close()
        log_print(f"Fraudulent transactions by age plot saved to {fraud_by_age_path}")

        # Plot fraud distribution by age group
        plt.figure(figsize=(8, 6))
        sns.countplot(x='age_group', hue='is_fraud', data=df_eda, palette='Set1')
        plt.title('Fraudulent Transactions by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        fraud_by_age_group_path = os.path.join(output_path, 'fraud_by_age_group.png')
        plt.savefig(fraud_by_age_group_path)
        plt.close()
        log_print(f"Fraudulent transactions by age group plot saved to {fraud_by_age_group_path}")

        # Plot the distribution of fraud by part of the day
        plt.figure(figsize=(8, 6))
        sns.countplot(x='part_of_day', hue='is_fraud', data=df_eda, palette='coolwarm')
        plt.title('Fraudulent Transactions by Part of Day')
        plt.xlabel('Part of Day')
        plt.ylabel('Count')
        fraud_by_part_of_day_path = os.path.join(output_path, 'fraud_by_part_of_day.png')
        plt.savefig(fraud_by_part_of_day_path)
        plt.close()
        log_print(f"Fraudulent transactions by part of day plot saved to {fraud_by_part_of_day_path}")

        # Group by 'transaction_hour' and calculate the sum of fraudulent transactions
        fraud_by_hour = df_eda.groupby('transaction_hour')['is_fraud'].sum()
        plt.figure(figsize=(8, 6))
        fraud_by_hour.plot(kind='line', color='purple')
        plt.title('Fraudulent Transactions by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Fraudulent Transactions')
        fraud_by_hour_path = os.path.join(output_path, 'fraud_by_hour.png')
        plt.savefig(fraud_by_hour_path)
        plt.close()
        log_print(f"Fraudulent transactions by hour plot saved to {fraud_by_hour_path}")

        # Bin the 'distance' column and analyze fraud by distance bins
        distance_bins = np.linspace(df_eda['distance'].min(), df_eda['distance'].max(), num=20)
        df_eda['distance_bin'] = pd.cut(df_eda['distance'], bins=distance_bins)
        fraud_by_distance_bin = df_eda.groupby('distance_bin')['is_fraud'].sum()
        plt.figure(figsize=(10, 6))
        fraud_by_distance_bin.plot(kind='bar', color='orange')
        plt.title('Fraudulent Transactions by Distance (Binned)')
        plt.xlabel('Distance Bins')
        plt.ylabel('Number of Fraudulent Transactions')
        plt.xticks(rotation=45)
        fraud_by_distance_path = os.path.join(output_path, 'fraud_by_distance.png')
        plt.savefig(fraud_by_distance_path)
        plt.close()
        log_print(f"Fraudulent transactions by distance plot saved to {fraud_by_distance_path}")

        fraud_by_city_pop_bin = df_eda.groupby('city_pop_bin')['is_fraud'].sum()


        # Plot the result as a bar plot
        plt.figure(figsize=(10, 6))
        fraud_by_city_pop_bin.plot(kind='bar', color='purple')
        plt.title('Fraudulent Transactions by City Population (Binned)')
        plt.xlabel('City Population Bins')
        plt.ylabel('Number of Fraudulent Transactions')
        plt.xticks(rotation=45)
        fraud_by_city_pop_path = os.path.join(output_path, 'fraud_by_city_pop.png')
        plt.savefig(fraud_by_city_pop_path)
        plt.close()
        log_print(f"Fraudulent transactions by city population plot saved to {fraud_by_city_pop_path}")

        # define numerical & categorical columns
        numerical_features = [
            feature for feature in df_eda.columns
            if pd.api.types.is_numeric_dtype(df_eda[feature]) and feature != "is_fraud"
        ]

        discrete_numerical_features = [feature for feature in numerical_features if df_eda[feature].nunique() < 25]

        continuous_numerical_features = [feature for feature in numerical_features if df_eda[feature].nunique() >= 25]

        categorical_features = [feature for feature in df_eda.columns if df_eda[feature].dtype == 'object']

        target_column = "is_fraud"
        log_print(f'There are {len(numerical_features)} numerical features : {numerical_features}')
        log_print(f'\nThere are {len(discrete_numerical_features)} discrete numerical features : {discrete_numerical_features}')
        log_print(f'\nThere are {len(continuous_numerical_features)} continuous numerical features : {continuous_numerical_features}')
        log_print(f'\nThere are {len(categorical_features)} categorical features : {categorical_features}')

        # Box plot for continuous numerical features

        # Calculate the number of rows required to plot all features
        num_cols = 4  # 4 plots per row
        num_rows = math.ceil(len(continuous_numerical_features) / num_cols)

        # Create a figure with subplots (4 plots per row)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * num_cols))

        # Flatten axes array for easy iteration (in case of multiple rows)
        axes = axes.flatten()

        # Iterate through numerical features and plot each one
        for i, feature in enumerate(continuous_numerical_features):
            sns.boxplot(
                y=df_eda[feature],
                ax=axes[i],
                notch=True,  # Add notch to the boxplot
                showcaps=False,  # Hide the caps
                flierprops={"marker": "x", "color": "r", "markersize": 8},  # Customizing outliers
                boxprops={"facecolor": (.3, .5, .7, .5)},  # Set the box color with transparency
                medianprops={"color": "r", "linewidth": 2},  # Customize the median line
                whiskerprops={"color": "black", "linewidth": 1.5},  # Customize whiskers
                width=0.4  # Shrink the box width
            )
            axes[i].set_title(f'Boxplot for {feature}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')

        # Hide any unused subplots if there are fewer than 4 features in the last row
        for i in range(len(continuous_numerical_features), len(axes)):
            axes[i].axis('off')

        # Adjust the layout to make sure all plots are well spaced
        plt.tight_layout()

        # Save the box plots
        boxplot_path = os.path.join(output_path, 'boxplots_continuous_features.png')
        plt.savefig(boxplot_path)
        plt.close()
        log_print(f"Box plots for continuous features saved to {boxplot_path}")

        # Distribution plot continuous numerical features
        # Calculate the number of rows required to plot all features in a 4x4 grid
        num_cols = 4  # 4 plots per row
        num_rows = math.ceil(len(continuous_numerical_features) / num_cols)

        # Create the subplots dynamically based on the number of numerical features
        plt.figure(figsize=(10, 3 * num_rows))  # Adjust figure size based on the number of rows

        # Loop through the numerical features to plot distribution plots
        for i, feature in enumerate(continuous_numerical_features, 1):
            plt.subplot(num_rows, num_cols, i)
            sns.kdeplot(df_eda[feature], fill=True, color='skyblue', edgecolor='black')
            plt.title(f'Distribution plot of {feature}', fontsize=8)
            plt.xlabel(feature)
            plt.ylabel('Density')

        # Adjust layout for better visibility
        plt.tight_layout()
        distribution_path = os.path.join(output_path, 'continuous_features_distribution.png')
        plt.savefig(distribution_path)
        plt.close()
        log_print(f"Distribution plots for continuous features saved to {distribution_path}")

        """The 'amt' column is highly skewed, so the Yeo-Johnson transformation will be applied.
        The 'city_pop' column is also skewed, but it will be removed. Instead, the categorical column 'city_pop_bin' will be used for the analysis.
        """

        # Check for Skewness in Numerical Columns
        for feature in continuous_numerical_features:
            skewness = df_eda[feature].skew()
            if abs(skewness) > 1:
                log_print(f"'{feature}' is highly skewed with skewness: {skewness:.2f}")

        """'amt' is highly skewed with skewness: 40.81
        'city_pop' is highly skewed with skewness: 5.59
        """

        # Check for Multicollinearity
        # Compute VIF for each numerical feature
        X = df_eda[continuous_numerical_features]
        X = add_constant(X)  # Add a constant column for VIF calculation
        vif = pd.DataFrame()
        vif['Feature'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_path = os.path.join(output_path, 'vif_report.txt')
        vif.to_csv(vif_path, index=False)
        log_print(f"VIF report saved to {vif_path}")

        # Apply Yeo-Johnson Transformation transformation
        transformation = PowerTransformer(method='yeo-johnson')
        df_eda['amt_yeo_johnson'] = transformation.fit_transform(df_eda[['amt']])

        # Plot original and transformed distributions side by side
        plt.figure(figsize=(14, 6))

        # Original Distribution
        plt.subplot(1, 2, 1)
        sns.histplot(df_eda['amt'], bins=50, kde=True, color='blue')
        plt.title('Original Distribution of Amount')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')

        # Transformed Distribution
        plt.subplot(1, 2, 2)
        sns.histplot(df_eda['amt_yeo_johnson'], bins=50, kde=True, color='green')
        plt.title('Yeo-Johnson-Transformed Distribution of Amount')
        plt.xlabel('Yeo-Johnson(Amount)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        yeo_johnson_path = os.path.join(output_path, 'yeo_johnson_transformation.png')
        plt.savefig(yeo_johnson_path)
        plt.close()
        log_print(f"Yeo-Johnson transformation plot saved to {yeo_johnson_path}")

        avg_amt_fraud_after_transformation = df_eda.groupby('is_fraud')['amt_yeo_johnson'].mean()

        # Plot the result
        avg_amt_fraud_trans_path = os.path.join(output_path, 'avg_amt_fraud_after_transformation.png')
        avg_amt_fraud_after_transformation.plot(kind='bar', color=['green', 'red'], figsize=(6, 4))
        plt.title('Average Transaction Amount by Fraud Status')
        plt.xlabel('Fraud Status')
        plt.ylabel('Average Amount')
        plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'], rotation=0)
        plt.savefig(avg_amt_fraud_trans_path)
        plt.close()
        log_print(f"Average transaction amount by fraud status plot saved to {avg_amt_fraud_trans_path}")



    print(f"EDA completed. Outputs are stored in {output_path}")


if __name__ == "__main__":
    # Example usage (replace with actual DataFrame input in the main script)
    print("This file is intended to be called from main.py.")
