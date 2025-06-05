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
                 max_depth: int = 100,           # Maximum depth of a tree
                 subsample: float = 0.8,       # Subsample ratio of the training instance
                 colsample_bytree: float = 0.8 # Subsample ratio of columns when constructing each tree
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
        )
        self.scaler = StandardScaler()
        self.numerical_features = []
        self.categorical_features = []
        self.fitted_feature_columns = None # Stores the exact order of all final features (numeric + OHE)

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        """
        Trains the XGBoost Classifier model.
        Dynamically identifies numeric and categorical features, applies one-hot encoding and scaling.
        """
        if user_features is None or item_features is None:
            raise ValueError("User and item features are required for a content-based recommender.")

        # Join Spark DataFrames and convert to Pandas for processing
        training_data_pd = log.join(user_features, on='user_idx', how='inner') \
                             .join(item_features, on='item_idx', how='inner') \
                             .drop('__iter') \
                             .toPandas()
        
        # Dynamically identify numerical features
        self.numerical_features = [
            col for col in training_data_pd.columns
            if col not in ['user_idx', 'item_idx', 'relevance']
            and pd.api.types.is_numeric_dtype(training_data_pd[col])
        ]
        
        # Dynamically identify categorical features based on object or categorical dtype
        self.categorical_features = [
            col for col in training_data_pd.columns
            if col not in ['user_idx', 'item_idx', 'relevance']
            and (pd.api.types.is_categorical_dtype(training_data_pd[col]) or pd.api.types.is_object_dtype(training_data_pd[col]))
        ]

        # Apply one-hot encoding for identified categorical features
        if self.categorical_features:
            training_data_pd = pd.get_dummies(training_data_pd, columns=self.categorical_features, drop_first=False)
        
        # Define features (X) and target (y)
        y = training_data_pd['relevance'] # Target variable
        
        # Select all features (original numerical ones + newly created OHE columns)
        all_features_to_model = [
            col for col in training_data_pd.columns
            if col not in ['user_idx', 'item_idx', 'relevance']
            and pd.api.types.is_numeric_dtype(training_data_pd[col]) # Ensure only numeric/OHE columns are selected
        ]
        
        X = training_data_pd[all_features_to_model]

        if X.empty:
            raise ValueError("No suitable features found for training after processing. Cannot train XGBoost.")

        # Store the exact order of ALL feature columns for consistency
        self.fitted_feature_columns = X.columns.tolist()

        # Fit the StandardScaler and transform the training features
        X_Scaled = self.scaler.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(X_Scaled, y, test_size=0.2, random_state=self.seed)
        self.model.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               early_stopping_rounds=50, # Stop if validation metric doesn't improve for 50 rounds
               verbose=False) # Set to True to see progress

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """
        Generates recommendations using the trained XGBoost Classifier model.
        Applies the same feature engineering (one-hot encoding) and scaling as during training.
        """
        if self.model is None:
            raise RuntimeError("The XGBoost Classifier model must be fitted using the 'fit' method before making predictions.")
        if user_features is None or item_features is None:
            raise ValueError("User and item features are required for prediction.")

        # 1. Create all possible user-item pairs (Spark DataFrame)
        all_pairs_spark = users.crossJoin(items)

        # 2. Join with user and item features (Spark DataFrame)
        prediction_data_spark = all_pairs_spark.join(user_features, on='user_idx', how='inner') \
                                             .join(item_features, on='item_idx', how='inner') \
                                             .drop('__iter')

        # Convert to Pandas DataFrame for scikit-learn processing
        prediction_data_pd = prediction_data_spark.toPandas()

        # Apply one-hot encoding for identified categorical features (same as in fit)
        if self.categorical_features:
            prediction_data_pd = pd.get_dummies(prediction_data_pd, columns=self.categorical_features, drop_first=False)
        
        # Select all relevant features for prediction
        all_features_for_predict = [
            col for col in prediction_data_pd.columns
            if col not in ['user_idx', 'item_idx', 'relevance']
            and pd.api.types.is_numeric_dtype(prediction_data_pd[col])
        ]
        
        # Create a new DataFrame for X_predict to avoid SettingWithCopyWarning
        X_predict_raw = prediction_data_pd[all_features_for_predict].copy()

        # Align ALL columns with the fitted model's features (numerical and OHE)
        missing_overall_cols = set(self.fitted_feature_columns) - set(X_predict_raw.columns)
        for col_name in missing_overall_cols:
            X_predict_raw[col_name] = 0 # Directly assign to the copy

        # Remove extra columns
        extra_overall_cols = set(X_predict_raw.columns) - set(self.fitted_feature_columns)
        if extra_overall_cols: # Only drop if there are columns to drop
            X_predict_raw = X_predict_raw.drop(columns=list(extra_overall_cols))
        
        # Ensure all columns are present and in the exact same order as during fit for the model
        X_predict = X_predict_raw[self.fitted_feature_columns]

        # Apply StandardScaler transform using the *fitted* scaler
        X_Scaled = self.scaler.transform(X_predict) 

        # Predict relevance scores using the XGBoost Classifier model
        # Use predict_proba for classification, and take the probability of the positive class (class 1)
        prediction_data_pd['relevance'] = self.model.predict_proba(X_Scaled)[:, 1]

        # Filter out already seen items if requested
        if filter_seen_items and log is not None:
            seen_items_pd = log.select("user_idx", "item_idx").distinct().toPandas()
            # For efficiency with larger datasets, use a set of tuples
            seen_items_set = set(tuple(row) for row in seen_items_pd[['user_idx', 'item_idx']].values)
            
            # Use a more efficient way to filter using .isin() or merge
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

        return recs_spark