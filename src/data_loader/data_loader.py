import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SnapDataLoader():
    def __init__(self, settings, paths):
        self.settings = settings
        self.paths = paths
        self.dataset_train = None
        self.dataset_test = None

    def load_data(self):
        self.dataset_train = pd.read_csv(self.paths.raw_dataset_path + 'train.csv')
        self.dataset_test = pd.read_csv(self.paths.raw_dataset_path + 'test.csv')

    def get_train_data(self):
        return self.dataset_train

    def get_test_data(self):
        return self.dataset_test


class EDA:
    def __init__(self, paths):
        self.train_df = pd.read_csv(paths.raw_dataset_path + 'train.csv')
        self.test_df = pd.read_csv(paths.raw_dataset_path + 'test.csv')
        self.paths = paths

    def load_and_preview(self):
        print("Train Data")
        print(self.train_df.head())
        print("\nTest Data")
        print(self.test_df.head())

    def check_missing_values(self):
        train_df = self.train_df
        test_df = self.test_df

        # Calculating missing value rate in percentage for each column in the train dataset
        train_missing_rate = train_df.isnull().mean() * 100
        train_missing_rate_df = pd.DataFrame(train_missing_rate, columns=['Train Missing Rate (%)'])

        # Calculating missing value rate in percentage for each column in the test dataset
        test_missing_rate = test_df.isnull().mean() * 100
        test_missing_rate_df = pd.DataFrame(test_missing_rate, columns=['Test Missing Rate (%)'])

        # Combining both DataFrames side by side
        missing_rate_df = pd.concat([train_missing_rate_df, test_missing_rate_df], axis=1)
        missing_rate_df.to_csv(self.paths.eda_output_path + 'missing_rates.csv')

        # Displaying the DataFrame
        print(missing_rate_df[missing_rate_df['Train Missing Rate (%)'] > 0])

    def summary_statistics(self):
        # Get summary statistics for numerical features in the train dataset
        print("Calculating summary statistics for features ...")
        # Transpose and calculate summary statistics for the train dataset
        train_summary = self.train_df.describe().T

        # Transpose and calculate summary statistics for the test dataset
        test_summary = self.test_df.describe().T

        # Concatenate the summaries with suffixes to distinguish between train and test
        combined_summary = pd.concat([train_summary, test_summary], axis=1,
                                     keys=['Train', 'Test'])
        # Flatten the MultiIndex columns and add proper suffixes
        combined_summary.columns = ['{}_{}'.format(col, src) for src, col in combined_summary.columns]

        # Save the combined summary to a CSV file
        combined_summary.to_csv(self.paths.eda_output_path + 'summary_statistics.csv')

        return combined_summary

    def plot_numerical_distributions(self, numerical_features, mode='train'):
        if mode == 'train':
            data = self.train_df
        else:
            data = self.test_df
        # Calculate the number of rows needed in the subplot grid
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols  # Ensures there are enough rows in the grid

        # Set up the matplotlib figure
        plt.figure(figsize=(n_cols * 6, n_rows * 4))  # Adjust overall figure size

        # Loop through each feature and create a subplot for its distribution
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data[feature][data[feature] > 0], color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')

        plt.tight_layout()  # Adjusts subplot params so that plots are nicely fit in the figure area

        # Check if the output path exists, if not, create it
        if not os.path.exists(self.paths.eda_output_path):
            os.makedirs(self.paths.eda_output_path)

        # Save the figure
        plt.savefig(os.path.join(self.paths.eda_output_path, f'numerical_distributions_{mode}.png'))
        plt.show()
        plt.close()  # Close the figure to free memory

    def plot_numerical_boxplots(self, numerical_features, mode='train'):
        if mode == 'train':
            data = self.train_df
        else:
            data = self.test_df
        # Calculate the number of rows needed in the subplot grid
        n_features = len(numerical_features)
        n_cols = 3  # Three columns for the grid
        n_rows = (n_features + n_cols - 1) // n_cols  # Ensure there are enough rows in the grid

        # Set up the matplotlib figure
        plt.figure(figsize=(n_cols * 6, n_rows * 4))  # Adjust overall figure size based on the number of features

        # Loop through each feature and create a subplot for its boxplot
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(x=data[feature][data[feature] > 0], color='lightgreen')
            plt.title(f'Boxplot of {feature}')
            plt.xlabel(feature)

        plt.tight_layout()  # Adjusts subplot parameters so that plots are nicely fit in the figure area

        # Check if the output path exists, if not, create it
        if not os.path.exists(self.paths.eda_output_path):
            os.makedirs(self.paths.eda_output_path)

        # Save the figure
        plt.savefig(os.path.join(self.paths.eda_output_path, f'boxplots_numerical_features_{mode}.png'))
        plt.show()
        plt.close()  # Close the figure to free memory

    def analyze_non_zero_distributions(self):
        # Define the columns to analyze
        columns = ['second_destination_final_price', 'round_ride_final_price']

        # Create a figure with 2 subplots
        plt.figure(figsize=(12, 6))

        for i, column in enumerate(columns, 1):
            # Filter out zeros
            non_zero_data = self.train_df[self.train_df[column] > 0][column]

            # Calculate the percentage of non-zero entries
            non_zero_percentage = 100 * len(non_zero_data) / len(self.train_df)
            print(f'Percentage of non-zero entries in {column}: {non_zero_percentage:.2f}%')

    def plot_categorical_distributions(self, categorical_features):

        # Calculate number of rows needed for subplots (with 3 columns)
        n_features = len(categorical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        # Create a figure with specified dimensions
        plt.figure(figsize=(n_cols * 6, n_rows * 4))

        for i, feature in enumerate(categorical_features):
            # Create subplot for each feature
            ax = plt.subplot(n_rows, n_cols, i + 1)
            sns.countplot(data=self.train_df, x=feature, hue=feature, palette='Set2', legend=False)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.xticks(rotation=45)  # Rotate x labels for better readability if necessary

            # Calculate the total number for normalization (to calculate percentages)
            total = float(len(self.train_df[feature]))

            # Annotate bars with percentage
            for p in ax.patches:
                height = p.get_height()  # Get the height of each bar
                # Calculate the percentage and format it
                percentage = '{:.1f}%'.format(100 * height / total) if total > 0 else '0%'
                # Annotate the percentage on top of each bar
                ax.text(p.get_x() + p.get_width() / 2., height, percentage, ha="center", va="bottom")

            # Calculate frequency table and print it
            freq_table = self.train_df[feature].value_counts()
            print(f"\nFrequency table for {feature}:\n{freq_table}\n")

        plt.tight_layout()  # Adjust subplots to fit into figure area nicely
        plt.savefig(os.path.join(self.paths.eda_output_path, 'categorical_distributions.png'))  # Save the figure
        plt.show()
        plt.close()  # Close the plot to free up memory

    def correlation_analysis(self, numerical_features, categorical_features):
        # Combine all features for correlation analysis
        all_features = numerical_features + categorical_features + ['day_of_week', 'hour_of_day']

        # Compute the correlation matrix
        corr_matrix = self.train_df[all_features].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.paths.eda_output_path + "/correlation.png")

        plt.show()

    def pointbiserial_correlation(self, numerical_features):
        # Assuming 'df' is your DataFrame and 'ride' is the binary target variable
        numerical_features = self.train_df[numerical_features+['day_of_week', 'hour_of_day']].columns.to_list()

        correlations = {}
        for feature in numerical_features:
            correlation, _ = pointbiserialr(self.train_df['ride (target)'], self.train_df[feature])
            correlations[feature] = correlation

        # Convert to DataFrame for better visualization
        correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation with Target'])
        print(correlation_df.sort_values(by='Correlation with Target', ascending=False))
        correlation_df.to_csv(self.paths.eda_output_path + "point_biserial_test.csv")

    def chi_square_test(self, categorical_features):
        chi_squared_stats = {}

        for feature in categorical_features:
            contingency_table = pd.crosstab(self.train_df['ride (target)'], self.train_df[feature])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            chi_squared_stats[feature] = (chi2, p)

        # Display the chi-square statistic and p-value for each feature
        for feature, stats in chi_squared_stats.items():
            print(f"{feature} - \t \t Chi-squared: {stats[0]}, \t \t  p-value: {stats[1]}")

    def correlation_with_target(self):
        # Numerical features correlation with the binary target
        numerical_features = [col for col in self.train_df.select_dtypes(include=[np.number]).columns if col != 'ride (target)']
        corr_dict = {}
        for feature in numerical_features:
            correlation, _ = pointbiserialr(self.train_df['ride (target)'], self.train_df[feature])
            corr_dict[feature] = correlation

        # Save numerical correlation results
        correlation_df = pd.DataFrame(list(corr_dict.items()), columns=['Feature', 'Correlation with Target'])
        correlation_df.to_csv(os.path.join(self.paths.eda_output_path, 'numerical_correlations.csv'), index=False)

        # Categorical features correlation using Logistic Regression
        categorical_features = ['waiting_time_enabled', 'for_friend_enabled', 'is_voucher_used', 'intercity',
                                'requested_service_type', 'in_hurry_enabled', 'treatment_group']
        for feature in categorical_features:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(self.train_df[feature])
            model = LogisticRegression()
            model.fit(encoded.reshape(-1, 1), self.train_df['ride (target)'])
            print(f"Correlation (Logistic Regression Coef) between {feature} and target: {model.coef_[0][0]}")

    def detect_outliers_with_dbscan(self, features, eps=0.5, min_samples=5):
        # Standardizing the data is crucial for clustering algorithms like DBSCAN
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.train_df[features])

        # Visualization using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        df_pca = pd.DataFrame(pca_result)

        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(df_pca)

        # Identify points classified as noise by DBSCAN (-1 label)
        self.train_df['outlier'] = clusters == -1

        # Count of identified outliers
        num_outliers = sum(self.train_df['outlier'])
        print(f"DBSCAN identified {num_outliers} outliers.")

        self.train_df['pca_one'] = pca_result[:, 0]
        self.train_df['pca_two'] = pca_result[:, 1]

        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='pca_one', y='pca_two', hue='outlier', palette={True: 'red', False: 'blue'},
                        data=self.train_df, legend='full', alpha=0.5)
        self.train_df.drop('pca_one', axis=1, inplace=True)
        self.train_df.drop('pca_two', axis=1, inplace=True)


        plt.title('PCA Plot of Data with Outliers Marked by DBSCAN')
        plt.savefig(self.paths.eda_output_path + "/dbscan_outliers_pca.png")
        plt.show()

        self.train_df.to_csv(self.paths.eda_output_path + "/dataset_with_outlayer.csv")
        self.train_df[self.train_df['outlier']==True]

        return self.train_df

    def plot_pca(self, data, features, file_name='pca'):
        # Standardizing the data is crucial for clustering algorithms like DBSCAN
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])

        # Visualization using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        data['pca_one'] = pca_result[:, 0]
        data['pca_two'] = pca_result[:, 1]

        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='pca_one', y='pca_two', hue=data['ride (target)'], palette={0: 'red', 1: 'green'},
                        data=data, legend='full', alpha=0.5)
        plt.title('PCA Plot of Data with Outliers Marked by DBSCAN')
        plt.savefig(self.paths.eda_output_path + f"/{file_name}.png")
        plt.show()

    def feature_engineering_suggestions(self):
        self.train_df['request_datetime'] = pd.to_datetime(self.train_df['request_datetime'])
        self.train_df['request_hour'] = self.train_df['request_datetime'].dt.hour
        self.train_df['request_day_of_week'] = self.train_df['request_datetime'].dt.dayofweek
        self.train_df['request_month'] = self.train_df['request_datetime'].dt.month

        print("Engineered Features")
        print(self.train_df[['request_hour', 'request_day_of_week', 'request_month']].head())

    def preprocess_datetime(self):
        # Convert 'request_datetime' to datetime
        self.train_df['request_datetime'] = pd.to_datetime(self.train_df['request_datetime'])

        # Extract day of the week and hour of the day
        self.train_df['day_of_week'] = self.train_df['request_datetime'].dt.dayofweek
        self.train_df['hour_of_day'] = self.train_df['request_datetime'].dt.hour
        self.train_df.drop('request_datetime', axis=1, inplace=True)

        self.test_df['request_datetime'] = pd.to_datetime(self.test_df['request_datetime'])

        # Extract day of the week and hour of the day
        self.test_df['day_of_week'] = self.test_df['request_datetime'].dt.dayofweek
        self.test_df['hour_of_day'] = self.test_df['request_datetime'].dt.hour
        self.test_df.drop('request_datetime', axis=1, inplace=True)

    def encode_categorical_features(self, categorical_features):
        encoder = LabelEncoder()
        for feature in categorical_features:
            self.train_df[feature] = encoder.fit_transform(self.train_df[feature])
            self.test_df[feature] = encoder.transform(self.test_df[feature])

    def run_eda(self):
        print("")
        self.load_and_preview()
        self.check_missing_values()
        self.summary_statistics()
        self.plot_numerical_distributions()
        self.plot_categorical_distributions()
        self.correlation_analysis()
        self.feature_engineering_suggestions()


