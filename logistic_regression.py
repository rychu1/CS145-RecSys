import pandas as pd
import numpy as np
from typing import Optional

from pyspark.sql import functions as sf
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sample_recommenders import BaseRecommender
from sim4rec.utils import pandas_to_spark
from pyspark.sql.types import LongType

class LogRegModel(BaseRecommender):
    """
    Custom recommender based on a scikit-learn Linear Regression model.
    Fits using only numeric features, without scaling or categorical encoding.
    """
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = LogisticRegression(max_iter = 1000, penalty = 'l1', solver = 'liblinear', C = 0.1)
        self.scaler = StandardScaler()
        self.numerical_features = [] # Will store the names of numerical features used for fitting
        self.categorical_features = [] # Will store the names of categorical features used for fitting

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        """
        Trains the scikit-learn Linear Regression model using only numeric features.
        Data is converted to Pandas for preprocessing and training.
        """
        if user_features is None or item_features is None:
            raise ValueError("User and item features are required for a content-based linear regression recommender.")

        # print("Starting scikit-learn Linear Regression model training (numeric features only)...")

        # Join Spark DataFrames and convert to Pandas for processing
        training_data_pd = log.join(user_features, on='user_idx', how='inner') \
                              .join(item_features, on='item_idx', how='inner') \
                              .drop('__iter') \
                              .toPandas()
        

        # Identify all numerical columns to be used as features
        # Exclude 'user_idx', 'item_idx', and 'relevance' (the target)
        self.numerical_features = [
            col for col in training_data_pd.columns
            if col not in ['user_idx', 'item_idx', 'relevance']
            and pd.api.types.is_numeric_dtype(training_data_pd[col])
        ]
        
        self.categorical_features = [
            col for col in training_data_pd.columns
            if col not in ['user_idx', 'item_idx', 'relevance']
            and pd.api.types.is_object_dtype(training_data_pd[col])
        ]
        
        if not self.numerical_features:
            raise ValueError("No numerical features.")
        if not self.categorical_features:
            raise ValueError("No categorical features.")

        X_cat = pd.get_dummies(training_data_pd[self.categorical_features], drop_first=True)
        X_num = training_data_pd[self.numerical_features]

        y = training_data_pd['relevance']  # Target variable
        X = pd.concat([X_cat, X_num], axis=1) # Select only the identified numerical features

        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)


    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:

        if self.model is None:
            raise RuntimeError("The Logistic Regression model must be fitted using the 'fit' method before making predictions.")
        if users is None or items is None:
            raise ValueError("User and item features are required for prediction.")

        # 1. Create all possible user-item pairs (Spark DataFrame)
        all_pairs_spark = users.crossJoin(items)

        # 2. Join with user and item features (Spark DataFrame)
        # This creates the full prediction data for the scikit-learn model
        prediction_data_spark = all_pairs_spark.drop('__iter')
        # Convert to Pandas DataFrame for scikit-learn processing
        prediction_data_pd = prediction_data_spark.toPandas()

        # print(prediction_data_pd.head(1))  # Display the first few rows for debugging
        # print (prediction_data_pd.columns)  # Display all columns for debugging


        # Select only the numerical features identified during fitting
        # This ensures the input to the model matches the training features
        X_cat = pd.get_dummies(prediction_data_pd[self.categorical_features], drop_first=True)
        X_num = prediction_data_pd[self.numerical_features]
        X_predict = pd.concat([X_cat, X_num], axis=1)

        X_scaled = self.scaler.transform(X_predict)
        # Predict relevance scores using the Logistic Regression model
        prediction_data_pd['relevance'] = self.model.predict(X_scaled)

        # Filter out already seen items if requested
        if filter_seen_items and log is not None:
            seen_items_pd = log.select("user_idx", "item_idx").distinct().toPandas()
            seen_items_set = set(tuple(row) for row in seen_items_pd[['user_idx', 'item_idx']].values)
            
            prediction_data_pd['is_seen'] = prediction_data_pd.apply(
                lambda row: (row['user_idx'], row['item_idx']) in seen_items_set, axis=1
            )
            prediction_data_pd = prediction_data_pd[~prediction_data_pd['is_seen']].drop(columns=['is_seen'])

        # Rank items by predicted relevance for each user
        prediction_data_pd = prediction_data_pd.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        top_k_recommendations_pd = prediction_data_pd.groupby('user_idx').head(k)

        # Select only the required columns (user_idx, item_idx, relevance) and convert back to Spark DataFrame
        recs_spark = pandas_to_spark(top_k_recommendations_pd[['user_idx', 'item_idx', 'relevance']])
        
        # Ensure correct types for the final Spark DataFrame
        recs_spark = recs_spark.withColumn("user_idx", sf.col("user_idx").cast(LongType())) \
                               .withColumn("item_idx", sf.col("item_idx").cast(LongType())) \
                               .withColumn("relevance", sf.col("relevance").cast("double"))

        # print("Recommendations generated.")
        return recs_spark
