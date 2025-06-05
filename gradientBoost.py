import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from pyspark.sql import DataFrame, functions as sf
from pyspark.sql.window import Window

# Import the Gradient Boosting model (XGBoost)
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from sample_recommenders import BaseRecommender
from sim4rec.utils import pandas_to_spark

class GradientBoostRecommender(BaseRecommender):
    """
    Custom recommender based on a scikit-learn compatible Gradient Boosting model (XGBoost).
    Dynamically identifies and uses both numeric and one-hot encoded categorical features.
    Applies StandardScaler to features.
    """
    def __init__(self, seed=None,
                 n_estimators: int = 100,      # Number of boosting rounds
                 learning_rate: float = 0.1,   # Step size shrinkage
                 max_depth: int = 6,           # Maximum depth of a tree
                 subsample: float = 0.8,       # Subsample ratio of the training instance
                 colsample_bytree: float = 0.8, # Subsample ratio of columns when constructing each tree,
                 alpha: float = 1           # L1 regularization term on weights
                ):
        super().__init__(seed=seed)
        # Initialize XGBoost Classifier model for binary relevance
        self.model = xgb.XGBClassifier(
            objective='binary:logistic', # Objective for binary classification
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.seed,
            n_jobs=-1, # Use all available cores
            tree_method='exact', # Faster algorithm for larger datasets
            alpha=alpha, # L1 regularization term on weights
            # lambda_=1, # L2 regularization term on weights
        )
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
            raise ValueError("No numerical features found after joining data. Cannot train Linear Regression.")
        if not self.categorical_features:
            raise ValueError("No categorical features found after joining data. Cannot train Linear Regression.")

        X_cat = pd.get_dummies(training_data_pd[self.categorical_features], drop_first=True)
        X_num = training_data_pd[self.numerical_features]
        # Define features (X) and target (y) for Linear Regression
        y = training_data_pd['relevance']  # Target variable
        X = pd.concat([X_cat, X_num], axis=1) # Select only the identified numerical features
        # print(f"Training data prepared with {len(X.columns)} features (numerical: {len(self.numerical_features)}, categorical: {len(self.categorical_features)})")
        # print(pd.DataFrame(X).head(1))

        self.scaler.fit(X)  # Fit the scaler to the training data (not used in this case, but kept for consistency)
        X_Scaled = self.scaler.transform(X)  # Apply scaling if needed, but here we are not scaling as per request
        
        # print(f"Training data prepared with {len(X.columns)} features (numerical: {len(self.numerical_features)}, categorical: {len(self.categorical_features)})")
        # Train the Linear Regression model
        self.model.fit(X_Scaled, y)

        # print("scikit-learn Linear Regression model trained successfully on numeric features.")

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """
        Generates recommendations using the trained scikit-learn Linear Regression model.
        Uses only numeric features for prediction.
        """
        if self.model is None:
            raise RuntimeError("The Linear Regression model must be fitted using the 'fit' method before making predictions.")
        if users is None or items is None:
            raise ValueError("User and item features are required for prediction.")

        # print(f"Generating top-{k} recommendations using scikit-learn Linear Regression (numeric features only)...")

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
        # self.scaler.fit(X_predict)  # Fit the scaler to the training data (not used in this case, but kept for consistency)
        X_Scaled = self.scaler.transform(X_predict) 

        # Predict relevance scores using the Linear Regression model
        prediction_data_pd['relevance'] = self.model.predict(X_Scaled)

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
        recs_spark = recs_spark.withColumn("user_idx", sf.col("user_idx").cast("int")) \
                               .withColumn("item_idx", sf.col("item_idx").cast("int")) \
                               .withColumn("relevance", sf.col("relevance").cast("double"))

        # print("Recommendations generated.")
        return recs_spark