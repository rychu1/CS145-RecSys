from recommenders.checkpoint3.graphCN import GraphCNRecommender
from recommenders.checkpoint3.lstm import MyLSTMRecommender
import pyspark.sql.functions as sf
from pyspark.sql.window import Window
class MODEL:
    """
    Hybrid recommender model combining GraphCN and MyLSTM recommenders.
    """
    model = None  # Recommender model instance
    weight= 1.0  # Weight for this model in hybrid recommendation
    
    def __init__(self,model=None, weight=1.0):
        self.model = model
        self.weight = weight

RECOMMENDERS = [
    MODEL(model=GraphCNRecommender(seed=42), weight=1.0),
    MODEL(model=MyLSTMRecommender(seed=42), weight=1.0)
]
class HybridRecommender:
    """
    Hybrid recommendation algorithm.
    Combines multiple recommendation algorithms.
    """
    
    def __init__(
        self,
        recommenders=[r.model for r in RECOMMENDERS],
        weights=[r.weight for r in RECOMMENDERS],
        seed=42
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            recommenders: List of recommender objects
            weights: Weights for each recommender (optional)
            seed: Random seed for reproducibility
        """
        self.recommenders = recommenders
        if weights is None:
            self.weights = [1.0 / len(recommenders)] * len(recommenders)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.seed = seed
        
    def fit(self, log, user_features=None, item_features=None):
        """
        Train all underlying recommenders.
        
        Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        for recommender in self.recommenders:
            recommender.fit(log=log, user_features=user_features, item_features=item_features)
            
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations by combining predictions from multiple recommenders.
        
        Args:
            log: Interaction log
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features (optional)
            item_features: Item features (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with combined relevance scores
        """
        # Generate recommendations from each recommender
        all_recs = []
        
        for i, (recommender, weight) in enumerate(zip(self.recommenders, self.weights)):
            # Generate recommendations
            recs = recommender.predict(
                log, 
                k=k*2,  # Request more recommendations to have more candidates
                users=users, 
                items=items,
                user_features=user_features,
                item_features=item_features,
                filter_seen_items=filter_seen_items
            )
            
            # Add recommender index and weight
            recs = recs.withColumn("recommender_idx", sf.lit(i))
            recs = recs.withColumn("weight", sf.lit(weight))
            
            all_recs.append(recs)
            
        # Combine all recommendations
        if not all_recs:
            return users.crossJoin(items).withColumn("relevance", sf.lit(0.5))
            
        combined_recs = all_recs[0]
        for recs in all_recs[1:]:
            combined_recs = combined_recs.unionByName(recs)
            
        # Aggregate relevance scores by user-item pair
        combined_recs = combined_recs.groupBy("user_idx", "item_idx").agg(
            sf.sum(sf.col("relevance") * sf.col("weight")).alias("relevance"),
            sf.collect_list("recommender_idx").alias("recommenders")
        )
        
        # Join back with item data
        result = combined_recs.join(items, on="item_idx")
        
        # Rank items by aggregated relevance for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        result = result.withColumn("rank", sf.row_number().over(window))
        
        # Filter top-k recommendations
        result = result.filter(sf.col("rank") <= k).drop("rank", "recommenders")
        
        return result
        
