"""
CSE 258 Assignment 2 - Recipe Recommendation System
Starter Template with Code Structure

This template provides the basic structure for your recommendation system project.
Follow the sections below and fill in the implementation details.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import ast

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data():
    """Load all necessary datasets"""
    print("Loading datasets...")
    
    # Load interaction splits
    train = pd.read_csv('interactions_train.csv')
    val = pd.read_csv('interactions_validation.csv')
    test = pd.read_csv('interactions_test.csv')
    
    # Load recipe data
    recipes = pd.read_csv('RAW_recipes.csv')
    
    # Load all interactions (for additional analysis)
    all_interactions = pd.read_csv('RAW_interactions.csv')
    
    print(f"Training interactions: {len(train):,}")
    print(f"Validation interactions: {len(val):,}")
    print(f"Test interactions: {len(test):,}")
    print(f"Total recipes: {len(recipes):,}")
    
    return train, val, test, recipes, all_interactions


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def extract_recipe_features(recipes_df):
    """Extract useful features from recipes"""
    features = recipes_df.copy()
    
    # Parse nutrition information
    # Format: [calories, total_fat, sugar, sodium, protein, saturated_fat, carbs]
    nutrition_cols = ['calories', 'total_fat', 'sugar', 'sodium', 
                     'protein', 'saturated_fat', 'carbs']
    
    nutrition_data = []
    for nutrition_str in features['nutrition']:
        try:
            nutrition_list = ast.literal_eval(nutrition_str)
            nutrition_data.append(nutrition_list)
        except:
            nutrition_data.append([np.nan] * 7)
    
    nutrition_df = pd.DataFrame(nutrition_data, columns=nutrition_cols)
    features = pd.concat([features, nutrition_df], axis=1)
    
    # Parse tags
    features['tags_list'] = features['tags'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    features['n_tags'] = features['tags_list'].apply(len)
    
    # Categorize by calorie level
    features['calorie_level'] = pd.cut(features['calories'], 
                                       bins=[0, 200, 400, 800, float('inf')],
                                       labels=['low', 'medium', 'high', 'very_high'])
    
    # Recipe complexity
    features['complexity_score'] = features['n_steps'] + features['n_ingredients'] * 0.5
    
    # Cooking time categories
    features['time_category'] = pd.cut(features['minutes'],
                                       bins=[0, 30, 60, 120, float('inf')],
                                       labels=['quick', 'moderate', 'long', 'very_long'])
    
    return features


def compute_recipe_popularity(interactions_df):
    """Compute popularity metrics for recipes"""
    popularity = interactions_df.groupby('recipe_id').agg({
        'rating': ['count', 'mean', 'std'],
        'user_id': 'count'
    }).reset_index()
    
    popularity.columns = ['recipe_id', 'n_ratings', 'avg_rating', 'std_rating', 'n_users']
    return popularity


def compute_user_statistics(interactions_df):
    """Compute statistics for each user"""
    user_stats = interactions_df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std'],
        'recipe_id': 'count'
    }).reset_index()
    
    user_stats.columns = ['user_id', 'n_ratings', 'avg_rating', 'std_rating', 'n_recipes']
    return user_stats


# ============================================================================
# 3. BASELINE MODELS
# ============================================================================

class GlobalMeanBaseline:
    """Predict global mean rating for all users and recipes"""
    
    def __init__(self):
        self.global_mean = None
    
    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        return self
    
    def predict(self, test_df):
        return np.full(len(test_df), self.global_mean)


class UserMeanBaseline:
    """Predict user's mean rating"""
    
    def __init__(self):
        self.user_means = {}
        self.global_mean = None
    
    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        self.user_means = train_df.groupby('user_id')['rating'].mean().to_dict()
        return self
    
    def predict(self, test_df):
        predictions = []
        for user_id in test_df['user_id']:
            pred = self.user_means.get(user_id, self.global_mean)
            predictions.append(pred)
        return np.array(predictions)


class RecipeMeanBaseline:
    """Predict recipe's mean rating"""
    
    def __init__(self):
        self.recipe_means = {}
        self.global_mean = None
    
    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        self.recipe_means = train_df.groupby('recipe_id')['rating'].mean().to_dict()
        return self
    
    def predict(self, test_df):
        predictions = []
        for recipe_id in test_df['recipe_id']:
            pred = self.recipe_means.get(recipe_id, self.global_mean)
            predictions.append(pred)
        return np.array(predictions)


class BiasBaseline:
    """User + Recipe bias model: prediction = global_mean + user_bias + recipe_bias"""
    
    def __init__(self):
        self.global_mean = None
        self.user_bias = {}
        self.recipe_bias = {}
    
    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        
        # Compute user biases
        user_means = train_df.groupby('user_id')['rating'].mean()
        self.user_bias = (user_means - self.global_mean).to_dict()
        
        # Compute recipe biases
        recipe_means = train_df.groupby('recipe_id')['rating'].mean()
        self.recipe_bias = (recipe_means - self.global_mean).to_dict()
        
        return self
    
    def predict(self, test_df):
        predictions = []
        for _, row in test_df.iterrows():
            pred = self.global_mean
            pred += self.user_bias.get(row['user_id'], 0)
            pred += self.recipe_bias.get(row['recipe_id'], 0)
            # Clip to valid rating range
            pred = np.clip(pred, 0, 5)
            predictions.append(pred)
        return np.array(predictions)


# ============================================================================
# 4. COLLABORATIVE FILTERING (Matrix Factorization)
# ============================================================================

class MatrixFactorization:
    """
    Simple Matrix Factorization using SGD
    Rating = user_factors @ recipe_factors.T
    
    TODO: Implement this using numpy or use a library like surprise/implicit
    """
    
    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.02, n_epochs=20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.recipe_factors = None
        self.global_mean = None
    
    def fit(self, train_df):
        """
        TODO: Implement SGD-based matrix factorization
        
        Steps:
        1. Initialize random user and recipe factor matrices
        2. For each epoch:
           - Shuffle training data
           - For each interaction (user, recipe, rating):
             * Compute prediction
             * Compute error
             * Update user and recipe factors using gradient descent
        3. Track training error
        """
        print("TODO: Implement Matrix Factorization")
        self.global_mean = train_df['rating'].mean()
        return self
    
    def predict(self, test_df):
        """
        TODO: Predict ratings by matrix multiplication
        prediction = user_factors[user_id] @ recipe_factors[recipe_id].T
        """
        # Placeholder
        return np.full(len(test_df), self.global_mean)


# ============================================================================
# 5. CONTENT-BASED FILTERING
# ============================================================================

def build_recipe_profiles(recipes_with_features):
    """
    Build content profiles for recipes using tags, ingredients, nutrition
    
    TODO: Create feature vectors for each recipe
    - One-hot encode popular tags
    - Normalize nutrition values
    - Encode cooking time, complexity, etc.
    """
    print("TODO: Implement recipe profile building")
    pass


def build_user_profiles(interactions, recipe_profiles):
    """
    Build user profiles based on recipes they've rated highly
    
    TODO: Aggregate recipe features weighted by ratings
    user_profile = weighted_average(recipe_features, weights=ratings)
    """
    print("TODO: Implement user profile building")
    pass


def content_based_recommendations(user_profile, recipe_profiles, top_k=10):
    """
    Recommend recipes similar to user's profile
    
    TODO: Compute similarity (cosine, euclidean) between user and recipes
    Return top K most similar recipes
    """
    print("TODO: Implement content-based recommendations")
    pass


# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================

def evaluate_rating_prediction(y_true, y_pred):
    """Evaluate rating prediction performance"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae
    }


def evaluate_ranking(recommendations_dict, test_interactions, k=10):
    """
    Evaluate recommendation quality using ranking metrics
    
    Args:
        recommendations_dict: {user_id: [list of recommended recipe_ids]}
        test_interactions: DataFrame with true interactions
        k: Number of top recommendations to consider
    
    Returns:
        Dictionary with Precision@K, Recall@K, NDCG@K
    """
    precisions = []
    recalls = []
    
    # Group test interactions by user
    test_by_user = test_interactions.groupby('user_id')['recipe_id'].apply(set).to_dict()
    
    for user_id, recommended in recommendations_dict.items():
        if user_id not in test_by_user:
            continue
        
        true_recipes = test_by_user[user_id]
        recommended_k = set(recommended[:k])
        
        if len(true_recipes) == 0:
            continue
        
        # Precision@K
        precision = len(recommended_k & true_recipes) / k
        precisions.append(precision)
        
        # Recall@K
        recall = len(recommended_k & true_recipes) / len(true_recipes)
        recalls.append(recall)
    
    return {
        f'Precision@{k}': np.mean(precisions) if precisions else 0,
        f'Recall@{k}': np.mean(recalls) if recalls else 0,
        f'F1@{k}': 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls)) 
                   if (precisions and recalls) else 0
    }


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("CSE 258 - Recipe Recommendation System")
    print("="*80)
    
    # 1. Load data
    train, val, test, recipes, all_interactions = load_data()
    
    # 2. Feature engineering
    print("\nExtracting features...")
    recipes_with_features = extract_recipe_features(recipes)
    recipe_popularity = compute_recipe_popularity(train)
    user_stats = compute_user_statistics(train)
    
    # 3. Train and evaluate baseline models
    print("\n" + "="*80)
    print("BASELINE MODELS")
    print("="*80)
    
    models = {
        'Global Mean': GlobalMeanBaseline(),
        'User Mean': UserMeanBaseline(),
        'Recipe Mean': RecipeMeanBaseline(),
        'User + Recipe Bias': BiasBaseline()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(train)
        
        # Predict on validation set
        val_predictions = model.predict(val)
        val_metrics = evaluate_rating_prediction(val['rating'].values, val_predictions)
        
        # Predict on test set
        test_predictions = model.predict(test)
        test_metrics = evaluate_rating_prediction(test['rating'].values, test_predictions)
        
        results[name] = {
            'validation': val_metrics,
            'test': test_metrics
        }
        
        print(f"  Validation - RMSE: {val_metrics['RMSE']:.4f}, MAE: {val_metrics['MAE']:.4f}")
        print(f"  Test       - RMSE: {test_metrics['RMSE']:.4f}, MAE: {test_metrics['MAE']:.4f}")
    
    # 4. Advanced models
    print("\n" + "="*80)
    print("ADVANCED MODELS")
    print("="*80)
    
    # TODO: Implement and train advanced models
    # - Matrix Factorization
    # - Content-Based Filtering
    # - Hybrid Models
    
    # 5. Generate recommendations
    print("\n" + "="*80)
    print("TOP-K RECOMMENDATIONS")
    print("="*80)
    
    # TODO: Generate top-K recommendations for test users
    # TODO: Evaluate using Precision@K, Recall@K, NDCG@K
    
    # 6. Cold-start analysis
    print("\n" + "="*80)
    print("COLD-START ANALYSIS")
    print("="*80)
    
    # TODO: Evaluate performance on users with few interactions
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

