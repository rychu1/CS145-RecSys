# %% [markdown]
# # Recommender Systems Analysis and Visualization
# This notebook performs an exploratory analysis of recommender systems using the Sim4Rec library.
# We'll generate synthetic data, compare multiple baseline recommenders, and visualize their performance.
# 

# %% [markdown]
# ## Cell: Import libraries and set up environment

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType, Optional

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


# Fix for KeyError: 'HOME' on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.expanduser("~")
if 'HADOOP_HOME' not in os.environ:
    #need to be c://hadoop on Windows
    os.environ['HADOOP_HOME'] = "C:\\hadoop"
# Set both driver and worker to use the current Python executable (should be 3.11)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# %% [markdown]
# ## Initialize Spark session

# %%
spark = (SparkSession.builder
    .appName("RecSysVisualization")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("Spark session initialized.")

# %% [markdown]
# ## Import competition modules 

# %%
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    SVMRecommender
)
from config import DEFAULT_CONFIG, EVALUATION_METRICS

print("Competition modules imported.")

# %% [markdown]
# ## Define custom recommender template
# Below is a template class for implementing a custom recommender system. You should extend this class with your own recommendation algorithm.

# %%
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

class MyRecommender(BaseRecommender):
    """
    Custom recommender based on a scikit-learn compatible Gradient Boosting model (XGBoost).
    Dynamically identifies and uses both numeric and one-hot encoded categorical features.
    Applies StandardScaler to features.
    """
    def __init__(self, seed=None,
                 n_estimators: int = 1,      # Number of boosting rounds
                 learning_rate: float = 0.1,   # Step size shrinkage
                 max_depth: int = 1,           # Maximum depth of a tree
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
            tree_method='hist', # Faster algorithm for larger datasets
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

# %% [markdown]
# ## Data Exploration Functions
# These functions help us understand the generated synthetic data.

# %%
def explore_user_data(users_df):
    """
    Explore user data distributions and characteristics.
    
    Args:
        users_df: DataFrame containing user data
    """
    print("=== User Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of users: {users_df.count()}")
    
    # User segments distribution
    segment_counts = users_df.groupBy("segment").count().toPandas()
    print("\nUser Segments Distribution:")
    for _, row in segment_counts.iterrows():
        print(f"  {row['segment']}: {row['count']} users ({row['count']/users_df.count()*100:.1f}%)")
    
    # Plot user segments
    plt.figure(figsize=(10, 6))
    plt.pie(segment_counts['count'], labels=segment_counts['segment'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('User Segments Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('user_segments_distribution.png')
    plt.show()
    print("User segments visualization saved to 'user_segments_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    users_pd = users_df.toPandas()
    
    # Analyze user feature distributions
    feature_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for segment in users_pd['segment'].unique():
                segment_data = users_pd[users_pd['segment'] == segment]
                plt.hist(segment_data[feature], alpha=0.5, bins=20, label=segment)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('user_feature_distributions.png')
        plt.show()
        print("User feature distributions saved to 'user_feature_distributions.png'")
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = users_pd[feature_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, annot=False, fmt='.2f'
        )
        plt.title('User Feature Correlations')
        plt.tight_layout()
        plt.savefig('user_feature_correlations.png')
        plt.show()
        print("User feature correlations saved to 'user_feature_correlations.png'")

def explore_item_data(items_df):
    """
    Explore item data distributions and characteristics.
    
    Args:
        items_df: DataFrame containing item data
    """
    print("\n=== Item Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of items: {items_df.count()}")
    
    # Item categories distribution
    category_counts = items_df.groupBy("category").count().toPandas()
    print("\nItem Categories Distribution:")
    for _, row in category_counts.iterrows():
        print(f"  {row['category']}: {row['count']} items ({row['count']/items_df.count()*100:.1f}%)")
    
    # Plot item categories
    plt.figure(figsize=(10, 6))
    plt.pie(category_counts['count'], labels=category_counts['category'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Item Categories Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('item_categories_distribution.png')
    plt.show()
    print("Item categories visualization saved to 'item_categories_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    items_pd = items_df.toPandas()
    
    # Analyze price distribution
    if 'price' in items_pd.columns:
        plt.figure(figsize=(14, 6))
        
        # Overall price distribution
        plt.subplot(1, 2, 1)
        plt.hist(items_pd['price'], bins=30, alpha=0.7)
        plt.title('Overall Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        
        # Price by category
        plt.subplot(1, 2, 2)
        for category in items_pd['category'].unique():
            category_data = items_pd[items_pd['category'] == category]
            plt.hist(category_data['price'], alpha=0.5, bins=20, label=category)
        plt.title('Price Distribution by Category')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('item_price_distributions.png')
        plt.show()
        print("Item price distributions saved to 'item_price_distributions.png'")
    
    # Analyze item feature distributions
    feature_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for category in items_pd['category'].unique():
                category_data = items_pd[items_pd['category'] == category]
                plt.hist(category_data[feature], alpha=0.5, bins=20, label=category)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('item_feature_distributions.png')
        plt.show()
        print("Item feature distributions saved to 'item_feature_distributions.png'")

def explore_interactions(history_df, users_df, items_df):
    """
    Explore interaction patterns between users and items.
    
    Args:
        history_df: DataFrame containing interaction history
        users_df: DataFrame containing user data
        items_df: DataFrame containing item data
    """
    print("\n=== Interaction Data Exploration ===")
    
    # Get basic statistics
    total_interactions = history_df.count()
    total_users = users_df.count()
    total_items = items_df.count()
    
    print(f"Total interactions: {total_interactions}")
    print(f"Interaction density: {total_interactions / (total_users * total_items) * 100:.4f}%")
    
    # Users with interactions
    users_with_interactions = history_df.select("user_idx").distinct().count()
    print(f"Users with at least one interaction: {users_with_interactions} ({users_with_interactions/total_users*100:.1f}%)")
    
    # Items with interactions
    items_with_interactions = history_df.select("item_idx").distinct().count()
    print(f"Items with at least one interaction: {items_with_interactions} ({items_with_interactions/total_items*100:.1f}%)")
    
    # Distribution of interactions per user
    interactions_per_user = history_df.groupBy("user_idx").count().toPandas()
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(interactions_per_user['count'], bins=20)
    plt.title('Distribution of Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    
    # Distribution of interactions per item
    interactions_per_item = history_df.groupBy("item_idx").count().toPandas()
    
    plt.subplot(1, 2, 2)
    plt.hist(interactions_per_item['count'], bins=20)
    plt.title('Distribution of Interactions per Item')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items')
    
    plt.tight_layout()
    plt.savefig('interaction_distributions.png')
    plt.show()
    print("Interaction distributions saved to 'interaction_distributions.png'")
    
    # Analyze relevance distribution
    if 'relevance' in history_df.columns:
        relevance_dist = history_df.groupBy("relevance").count().toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.bar(relevance_dist['relevance'].astype(str), relevance_dist['count'])
        plt.title('Distribution of Relevance Scores')
        plt.xlabel('Relevance Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('relevance_distribution.png')
        plt.show()
        print("Relevance distribution saved to 'relevance_distribution.png'")
    
    # If we have user segments and item categories, analyze cross-interactions
    if 'segment' in users_df.columns and 'category' in items_df.columns:
        # Join with user segments and item categories
        interaction_analysis = (history_df
            .join(users_df.select('user_idx', 'segment'), on='user_idx')
            .join(items_df.select('item_idx', 'category'), on='item_idx')
        )
        
        # Count interactions by segment and category
        segment_category_counts = interaction_analysis.groupBy('segment', 'category').count().toPandas()
        
        # Create a pivot table
        pivot_table = segment_category_counts.pivot(index='segment', columns='category', values='count').fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='g', cmap='viridis')
        plt.title('Interactions Between User Segments and Item Categories')
        plt.tight_layout()
        plt.savefig('segment_category_interactions.png')
        plt.show()
        print("Segment-category interactions saved to 'segment_category_interactions.png'")

print("Data exploration functions defined.")

# %% [markdown]
# ## Performance Visualization Functions
# These functions create visualizations for comparing recommender performance.

# %%
def visualize_recommender_performance(results_df, recommender_names):
    """
    Visualize the performance of recommenders in terms of revenue and key metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    plt.figure(figsize=(16, 16))
    
    # Plot total revenue comparison
    plt.subplot(3, 2, 1)
    x = np.arange(len(recommender_names))
    width = 0.35
    plt.bar(x - width/2, results_df['train_total_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_total_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot average revenue per iteration
    plt.subplot(3, 2, 2)
    plt.bar(x - width/2, results_df['train_avg_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_avg_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Avg Revenue per Iteration')
    plt.title('Average Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot discounted revenue comparison (if available)
    plt.subplot(3, 2, 3)
    if 'train_discounted_revenue' in results_df.columns and 'test_discounted_revenue' in results_df.columns:
        plt.bar(x - width/2, results_df['train_discounted_revenue'], width, label='Training')
        plt.bar(x + width/2, results_df['test_discounted_revenue'], width, label='Testing')
        plt.xlabel('Recommender')
        plt.ylabel('Avg Discounted Revenue')
        plt.title('Discounted Revenue Comparison')
        plt.xticks(x, results_df['name'])
        plt.legend()
    
    # Plot revenue trajectories
    plt.subplot(3, 2, 4)
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(results_df['name']):
        # Combined train and test trajectories
        train_revenue = results_df.iloc[i]['train_revenue']
        test_revenue = results_df.iloc[i]['test_revenue']
        
        # Ensure they are lists
        if isinstance(train_revenue, (float, np.floating, int, np.integer)):
            train_revenue = [train_revenue]
        if isinstance(test_revenue, (float, np.floating, int, np.integer)):
            test_revenue = [test_revenue]
        
        iterations = list(range(len(train_revenue))) + list(range(len(test_revenue)))
        revenues = train_revenue + test_revenue
        
        plt.plot(iterations, revenues, marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], label=name)
        
        # Add a vertical line to separate train and test
        if i == 0:  # Only add the line once
            plt.axvline(x=len(train_revenue)-0.5, color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
    
    plt.xlabel('Iteration')
    plt.ylabel('Revenue')
    plt.title('Revenue Trajectory (Training → Testing)')
    plt.legend()
    
    # Plot ranking metrics comparison - Training
    plt.subplot(3, 2, 5)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'train_{m}' in results_df.columns]
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Extended palette
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'train_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Training Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    # Plot ranking metrics comparison - Testing
    plt.subplot(3, 2, 6)
    
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'test_{m}' in results_df.columns]
    
    # Get best-performing model (by test total revenue) to highlight
    best_model_idx = results_df['test_total_revenue'].idxmax()
    best_model_name = results_df.iloc[best_model_idx]['name']
    
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'test_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)],
                alpha=0.7 if model_name != best_model_name else 1.0)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Test Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('recommender_performance_comparison.png')
    plt.show()
    print("\nPerformance visualizations saved to 'recommender_performance_comparison.png'")

def visualize_detailed_metrics(results_df, recommender_names):
    """
    Create detailed visualizations for each metric and recommender.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    # Create a figure for metric trajectories
    plt.figure(figsize=(16, 16))
    
    # Get all available metrics
    all_metrics = []
    if len(results_df) > 0 and 'train_metrics' in results_df.columns:
        first_train_metrics = results_df.iloc[0]['train_metrics'][0]
        all_metrics = list(first_train_metrics.keys())
    
    # Select key metrics to visualize
    key_metrics = ['revenue', 'discounted_revenue', 'precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Plot metric trajectories for each key metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']
    
    for i, metric in enumerate(key_metrics):
        if i < 6:  # Limit to 6 metrics to avoid overcrowding
            plt.subplot(3, 2, i+1)
            
            for j, name in enumerate(results_df['name']):
                row = results_df[results_df['name'] == name].iloc[0]
                
                # Get metric values for training phase
                train_values = []
                for train_metric in row['train_metrics']:
                    if metric in train_metric:
                        train_values.append(train_metric[metric])
                
                # Get metric values for testing phase
                test_values = []
                for test_metric in row['test_metrics']:
                    if metric in test_metric:
                        test_values.append(test_metric[metric])
                
                # Plot training phase
                plt.plot(
                    range(len(train_values)),
                    train_values,
                    marker=markers[j % len(markers)],
                    color=colors[j % len(colors)],
                    linestyle='-',
                    label=f"{name} (train)"
                )
                
                # Plot testing phase
                plt.plot(
                    range(len(train_values), len(train_values) + len(test_values)),
                    test_values,
                    marker=markers[j % len(markers)],
                    color=colors[j % len(colors)],
                    linestyle='--',
                    label=f"{name} (test)"
                )
                
                # Add a vertical line to separate train and test
                if j == 0:  # Only add the line once
                    plt.axvline(x=len(train_values) - 0.5, color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
            
            # Get metric info from EVALUATION_METRICS if available
            if metric in EVALUATION_METRICS:
                metric_info = EVALUATION_METRICS[metric]
                metric_name = metric_info['name']
                plt.title(f"{metric_name} Trajectory")
            else:
                plt.title(f"{metric.replace('_', ' ').title()} Trajectory")
            
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            
            # Add legend to the last plot or the final one in layout
            if i == len(key_metrics) - 1 or i == 5:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('recommender_metrics_trajectories.png')
    plt.show()
    print("Detailed metrics visualizations saved to 'recommender_metrics_trajectories.png'")
    
    # Create a correlation heatmap of metrics
    plt.figure(figsize=(14, 12))
    
    # Extract metrics columns
    metric_cols = [col for col in results_df.columns if col.startswith('train_') or col.startswith('test_')]
    metric_cols = [col for col in metric_cols if not col.endswith('_metrics') and not col.endswith('_revenue')]
    
    if len(metric_cols) > 1:
        correlation_df = results_df[metric_cols].corr()
        # Plot heatmap
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig('metrics_correlation_heatmap.png')
        plt.show()
        print("Metrics correlation heatmap saved to 'metrics_correlation_heatmap.png'")

def calculate_discounted_cumulative_gain(recommendations, k=5, discount_factor=0.85):
    """
    Calculate the Discounted Cumulative Gain for recommendations.
    
    Args:
        recommendations: DataFrame with recommendations (must have relevance column)
        k: Number of items to consider
        discount_factor: Factor to discount gains by position
    
    Returns:
        float: Average DCG across all users
    """
    # Group by user and calculate per-user DCG
    user_dcg = []
    grouped_recs = (recommendations
        .groupBy("user_idx")
        .agg(sf.collect_list(sf.struct("relevance", "rank")).alias("recommendations"))
    ).collect()
    
    for row in grouped_recs:
        user_id = row['user_idx']
        user_rec_list = sorted(row['recommendations'], key=lambda x: x[1])
        
        dcg = 0
        for i, rec in enumerate(user_rec_list[:k]):
            rel = rec['relevance']
            # Apply discount based on position
            dcg += rel * (discount_factor ** i)
        user_dcg.append(dcg)
    
    return np.mean(user_dcg) if user_dcg else 0.0

print("Performance visualization functions defined.")

# %% [markdown]
# ## Generate Data and Perform Exploratory Analysis

# %%
# We'll create a smaller dataset for demonstration.
config = DEFAULT_CONFIG.copy()
config['data_generation']['n_users'] = 1000  # Reduced from 10,000
config['data_generation']['n_items'] = 200   # Reduced from 1,000
config['data_generation']['seed'] = 29       # Fixed seed for reproducibility

# Get train-test split parameters
train_iterations = config['simulation']['train_iterations']
test_iterations = config['simulation']['test_iterations']

print(f"Running train-test simulation with {train_iterations} training iterations and {test_iterations} testing iterations")

# Initialize data generator
data_generator = CompetitionDataGenerator(
    spark_session=spark,
    **config['data_generation']
)

# Generate user data
users_df = data_generator.generate_users()
print(f"Generated {users_df.count()} users")

# Generate item data
items_df = data_generator.generate_items()
print(f"Generated {items_df.count()} items")

# Generate initial interaction history
history_df = data_generator.generate_initial_history(
    config['data_generation']['initial_history_density']
)
print(f"Generated {history_df.count()} initial interactions")

print("\n=== Starting Exploratory Data Analysis ===")
# explore_user_data(users_df)
# explore_item_data(items_df)
# explore_interactions(history_df, users_df, items_df)

print("Exploratory analysis complete.")

# %% [markdown]
# ## Set up Generators and Recommenders

# %%
# Prepare data generators for the simulator
user_generator, item_generator = data_generator.setup_data_generators()
from gradientBoost import GradientBoostRecommender
from linearRegression import LinearRegressionRecommender
# Initialize the recommenders we want to compare
recommenders = [
    RandomRecommender(seed=config['data_generation']['seed']),
    PopularityRecommender(alpha=1.0, seed=config['data_generation']['seed']),
    ContentBasedRecommender(similarity_threshold=0.0, seed=config['data_generation']['seed']),
    # MyRecommender(seed=config['data_generation']['seed']),  # Custom template class
    # LinearRegressionRecommender(seed=config['data_generation']['seed']),
    GradientBoostRecommender(seed=config['data_generation']['seed']),
]
recommender_names = [
    "Random",
    "Popularity",
    "ContentBased",
    # "MyRecommender",
    "LinearRegression",
    "GradientBoost"
    ]
# Fit each recommender on the initial history
for recommender in recommenders:
    recommender.fit(log=data_generator.history_df, user_features=data_generator.users_df, item_features=data_generator.items_df)

print("Recommenders set up and initial fit complete.")

# %% [markdown]
# ## Train-Test Simulation and Evaluation

# %%
import pandas as pd

results = []

for name, recommender in zip(recommender_names, recommenders):
    print(f"\nEvaluating {name}:")
    
    # Clean up any existing simulator data directory for this recommender
    simulator_data_dir = f"simulator_train_test_data_{name}"
    if os.path.exists(simulator_data_dir):
        shutil.rmtree(simulator_data_dir)
        print(f"Removed existing simulator data directory: {simulator_data_dir}")
    
    # Ensure 'relevance' column is LongType for compatibility
    from pyspark.sql.types import LongType

    history_df_casted = data_generator.history_df.withColumn(
        "relevance", data_generator.history_df["relevance"].cast(LongType())
    )

    # Initialize simulator
    simulator = CompetitionSimulator(
        user_generator=user_generator,
        item_generator=item_generator,
        data_dir=simulator_data_dir,
        log_df=history_df_casted,  # Use casted DataFrame
        conversion_noise_mean=config['simulation']['conversion_noise_mean'],
        conversion_noise_std=config['simulation']['conversion_noise_std'],
        spark_session=spark,
        seed=config['data_generation']['seed']
    )
    
    # Run simulation with train-test split
    train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
        recommender=recommender,
        train_iterations=train_iterations,
        test_iterations=test_iterations,
        user_frac=config['simulation']['user_fraction'],
        k=config['simulation']['k'],
        filter_seen_items=config['simulation']['filter_seen_items'],
        retrain=config['simulation']['retrain']
    )
    
    # Calculate average metrics
    train_avg_metrics = {}
    for metric_name in train_metrics[0].keys():
        values = [m[metric_name] for m in train_metrics]
        train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
    
    test_avg_metrics = {}
    for metric_name in test_metrics[0].keys():
        values = [m[metric_name] for m in test_metrics]
        test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
    
    # Store results
    results.append({
        "name": name,
        "train_total_revenue": sum(train_revenue),
        "test_total_revenue": sum(test_revenue),
        "train_avg_revenue": np.mean(train_revenue),
        "test_avg_revenue": np.mean(test_revenue),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_revenue": train_revenue,
        "test_revenue": test_revenue,
        **train_avg_metrics,
        **test_avg_metrics
    })
    
    # Print summary for this recommender
    print(f"  Training Phase - Total Revenue: {sum(train_revenue):.2f}")
    print(f"  Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
    performance_change = ((sum(test_revenue) / len(test_revenue)) / (sum(train_revenue) / len(train_revenue)) - 1) * 100
    print(f"  Performance Change: {performance_change:.2f}%")

print("\n=== Simulation and evaluation complete for all recommenders. ===")

# Convert to DataFrame for easy comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)

# Print summary table
print("\nRecommender Evaluation Results (sorted by test revenue):")
summary_cols = [
    "name", "train_total_revenue", "test_total_revenue", 
    "train_avg_revenue", "test_avg_revenue",
    "train_precision_at_k", "test_precision_at_k",
    "train_ndcg_at_k", "test_ndcg_at_k",
    "train_mrr", "test_mrr",
    "train_discounted_revenue", "test_discounted_revenue"
]
summary_cols = [col for col in summary_cols if col in results_df.columns]

if len(summary_cols) > 0:
    print(results_df[summary_cols].to_string(index=False))
else:
    print("No summary columns to display.")

# %% [markdown]
# ## Results Visualization

# %%
print("\n=== Visualizing Recommender Performance ===")
visualize_recommender_performance(results_df, recommender_names)
visualize_detailed_metrics(results_df, recommender_names)

print("\nVisualization complete.")

# %%



