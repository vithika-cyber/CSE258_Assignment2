"""
CSE 258 Assignment 2 - Advanced Recommendation Models
Implements Matrix Factorization, Content-Based, and Hybrid approaches
"""

import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, SVDpp, NMF, Dataset, Reader
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RECIPE RECOMMENDATION - ADVANCED MODELS")
print("="*80)

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """Load all datasets and perform preprocessing"""
    print("\n[1/8] Loading datasets...")
    
    train = pd.read_csv('interactions_train.csv')
    val = pd.read_csv('interactions_validation.csv')
    test = pd.read_csv('interactions_test.csv')
    recipes = pd.read_csv('RAW_recipes.csv')
    
    print(f"  Training: {len(train):,} interactions")
    print(f"  Validation: {len(val):,} interactions")
    print(f"  Test: {len(test):,} interactions")
    print(f"  Recipes: {len(recipes):,}")
    
    return train, val, test, recipes


def extract_recipe_features(recipes_df):
    """Extract and normalize recipe features"""
    print("\n[2/8] Extracting recipe features...")
    
    features = recipes_df.copy()
    
    # Parse nutrition information
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
    
    # Fill missing nutrition values with median
    for col in nutrition_cols:
        features[col].fillna(features[col].median(), inplace=True)
    
    # Parse tags
    features['tags_list'] = features['tags'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    features['tags_str'] = features['tags_list'].apply(lambda x: ' '.join(x))
    
    # Recipe complexity features
    features['complexity_score'] = features['n_steps'] + features['n_ingredients'] * 0.5
    
    # Handle outliers in minutes (cap at 500 minutes)
    features['minutes_capped'] = features['minutes'].clip(upper=500)
    
    print(f"  Extracted features for {len(features):,} recipes")
    return features


# ============================================================================
# 2. BASELINE MODELS
# ============================================================================

class GlobalMeanBaseline:
    def __init__(self):
        self.global_mean = None
    
    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        return self
    
    def predict(self, test_df):
        return np.full(len(test_df), self.global_mean)


class BiasBaseline:
    def __init__(self, alpha=5):
        self.global_mean = None
        self.user_bias = {}
        self.recipe_bias = {}
        self.alpha = alpha  # Regularization
    
    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()
        
        # Regularized user biases
        user_ratings = train_df.groupby('user_id')['rating']
        user_means = user_ratings.mean()
        user_counts = user_ratings.count()
        self.user_bias = ((user_means - self.global_mean) * user_counts / 
                         (user_counts + self.alpha)).to_dict()
        
        # Regularized recipe biases
        recipe_ratings = train_df.groupby('recipe_id')['rating']
        recipe_means = recipe_ratings.mean()
        recipe_counts = recipe_ratings.count()
        self.recipe_bias = ((recipe_means - self.global_mean) * recipe_counts / 
                           (recipe_counts + self.alpha)).to_dict()
        
        return self
    
    def predict(self, test_df):
        predictions = []
        for _, row in test_df.iterrows():
            pred = self.global_mean
            pred += self.user_bias.get(row['user_id'], 0)
            pred += self.recipe_bias.get(row['recipe_id'], 0)
            pred = np.clip(pred, 0, 5)
            predictions.append(pred)
        return np.array(predictions)


# ============================================================================
# 3. MATRIX FACTORIZATION (Collaborative Filtering)
# ============================================================================

class MatrixFactorizationSVD:
    """Matrix Factorization using Surprise's SVD"""
    
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.model = None
        self.trainset = None
    
    def fit(self, train_df):
        print(f"    Training SVD (factors={self.n_factors}, epochs={self.n_epochs})...")
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(
            train_df[['user_id', 'recipe_id', 'rating']], 
            reader
        )
        self.trainset = data.build_full_trainset()
        
        # Train SVD model
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr,
            reg_all=self.reg,
            verbose=False
        )
        self.model.fit(self.trainset)
        
        return self
    
    def predict(self, test_df):
        predictions = []
        for _, row in test_df.iterrows():
            pred = self.model.predict(
                row['user_id'], 
                row['recipe_id']
            ).est
            predictions.append(pred)
        return np.array(predictions)


class MatrixFactorizationSVDpp:
    """SVD++ - SVD with implicit feedback"""
    
    def __init__(self, n_factors=20, n_epochs=20):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.model = None
        self.trainset = None
    
    def fit(self, train_df):
        print(f"    Training SVD++ (factors={self.n_factors}, epochs={self.n_epochs})...")
        
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(
            train_df[['user_id', 'recipe_id', 'rating']], 
            reader
        )
        self.trainset = data.build_full_trainset()
        
        # SVD++ model (slower but considers implicit feedback)
        self.model = SVDpp(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            verbose=False
        )
        self.model.fit(self.trainset)
        
        return self
    
    def predict(self, test_df):
        predictions = []
        for _, row in test_df.iterrows():
            pred = self.model.predict(
                row['user_id'], 
                row['recipe_id']
            ).est
            predictions.append(pred)
        return np.array(predictions)


# ============================================================================
# 4. CONTENT-BASED FILTERING
# ============================================================================

class ContentBasedRecommender:
    """Content-based filtering using recipe features"""
    
    def __init__(self, recipes_df):
        self.recipes_df = recipes_df
        self.recipe_features = None
        self.user_profiles = None
        self.global_mean = None
        self.recipe_id_to_idx = {}
        self.idx_to_recipe_id = {}
    
    def build_recipe_features(self):
        """Build feature matrix for recipes"""
        print("    Building recipe feature matrix...")
        
        # Create recipe ID mapping
        self.recipe_id_to_idx = {
            rid: idx for idx, rid in enumerate(self.recipes_df['id'])
        }
        self.idx_to_recipe_id = {
            idx: rid for rid, idx in self.recipe_id_to_idx.items()
        }
        
        # TF-IDF on tags
        tfidf = TfidfVectorizer(max_features=100)
        tags_features = tfidf.fit_transform(self.recipes_df['tags_str']).toarray()
        
        # Normalize nutrition and complexity features
        scaler = StandardScaler()
        numeric_features = scaler.fit_transform(
            self.recipes_df[[
                'calories', 'protein', 'total_fat', 'carbs',
                'n_steps', 'n_ingredients', 'minutes_capped'
            ]]
        )
        
        # Combine features
        self.recipe_features = np.hstack([tags_features, numeric_features])
        
        print(f"    Recipe feature matrix: {self.recipe_features.shape}")
        return self
    
    def build_user_profiles(self, train_df):
        """Build user profiles from their rating history"""
        print("    Building user profiles...")
        
        self.global_mean = train_df['rating'].mean()
        self.user_profiles = {}
        
        # Group by user
        for user_id, group in train_df.groupby('user_id'):
            # Get recipes rated by this user
            recipe_ids = group['recipe_id'].values
            ratings = group['rating'].values
            
            # Only use highly rated recipes (>= 4) for profile
            high_rated_mask = ratings >= 4
            
            if high_rated_mask.sum() == 0:
                continue
            
            # Get feature indices
            feature_indices = [
                self.recipe_id_to_idx[rid] 
                for rid in recipe_ids[high_rated_mask]
                if rid in self.recipe_id_to_idx
            ]
            
            if len(feature_indices) == 0:
                continue
            
            # User profile = mean of highly rated recipe features
            user_feature_matrix = self.recipe_features[feature_indices]
            user_profile = user_feature_matrix.mean(axis=0)
            self.user_profiles[user_id] = user_profile
        
        print(f"    Built profiles for {len(self.user_profiles):,} users")
        return self
    
    def fit(self, train_df):
        self.build_recipe_features()
        self.build_user_profiles(train_df)
        return self
    
    def predict(self, test_df):
        predictions = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            recipe_id = row['recipe_id']
            
            # Default prediction
            pred = self.global_mean
            
            # If user has profile and recipe exists
            if user_id in self.user_profiles and recipe_id in self.recipe_id_to_idx:
                user_profile = self.user_profiles[user_id]
                recipe_idx = self.recipe_id_to_idx[recipe_id]
                recipe_feature = self.recipe_features[recipe_idx]
                
                # Compute similarity (cosine)
                similarity = np.dot(user_profile, recipe_feature) / (
                    np.linalg.norm(user_profile) * np.linalg.norm(recipe_feature) + 1e-10
                )
                
                # Scale similarity to rating range
                pred = self.global_mean + similarity * 2
                pred = np.clip(pred, 0, 5)
            
            predictions.append(pred)
        
        return np.array(predictions)


# ============================================================================
# 5. HYBRID MODEL
# ============================================================================

class HybridRecommender:
    """Hybrid model combining CF and Content-Based"""
    
    def __init__(self, cf_model, cb_model, alpha=0.7):
        self.cf_model = cf_model  # Collaborative filtering model
        self.cb_model = cb_model  # Content-based model
        self.alpha = alpha  # Weight for CF (1-alpha for CB)
    
    def fit(self, train_df):
        print(f"    Training Hybrid (alpha={self.alpha})...")
        print("    Training CF component...")
        self.cf_model.fit(train_df)
        print("    Training CB component...")
        self.cb_model.fit(train_df)
        return self
    
    def predict(self, test_df):
        cf_predictions = self.cf_model.predict(test_df)
        cb_predictions = self.cb_model.predict(test_df)
        
        # Weighted combination
        hybrid_predictions = (self.alpha * cf_predictions + 
                             (1 - self.alpha) * cb_predictions)
        
        return np.clip(hybrid_predictions, 0, 5)


# ============================================================================
# 6. EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, train_df, val_df, test_df, model_name):
    """Evaluate model on validation and test sets"""
    print(f"\n  Evaluating {model_name}...")
    
    # Validation performance
    val_pred = model.predict(val_df)
    val_rmse = np.sqrt(mean_squared_error(val_df['rating'], val_pred))
    val_mae = mean_absolute_error(val_df['rating'], val_pred)
    
    # Test performance
    test_pred = model.predict(test_df)
    test_rmse = np.sqrt(mean_squared_error(test_df['rating'], test_pred))
    test_mae = mean_absolute_error(test_df['rating'], test_pred)
    
    print(f"    Validation: RMSE={val_rmse:.4f}, MAE={val_mae:.4f}")
    print(f"    Test:       RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
    
    return {
        'model': model_name,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'predictions': test_pred
    }


def cold_start_analysis(model, train_df, test_df, model_name):
    """Analyze model performance on cold-start users"""
    print(f"\n  Cold-start analysis for {model_name}...")
    
    # Count user interactions in training
    user_counts = train_df.groupby('user_id').size()
    
    # Categorize test users
    results = {}
    categories = {
        'Cold (1-3)': (1, 3),
        'Warm (4-10)': (4, 10),
        'Hot (11+)': (11, float('inf'))
    }
    
    for cat_name, (min_count, max_count) in categories.items():
        # Filter users in this category
        users_in_cat = user_counts[
            (user_counts >= min_count) & (user_counts <= max_count)
        ].index
        
        test_subset = test_df[test_df['user_id'].isin(users_in_cat)]
        
        if len(test_subset) == 0:
            continue
        
        # Evaluate on this subset
        pred = model.predict(test_subset)
        rmse = np.sqrt(mean_squared_error(test_subset['rating'], pred))
        mae = mean_absolute_error(test_subset['rating'], pred)
        
        results[cat_name] = {
            'n_users': len(users_in_cat),
            'n_test': len(test_subset),
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"    {cat_name}: {len(test_subset)} test samples, "
              f"RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return results


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_results(results_df, cold_start_results):
    """Create visualizations of results"""
    print("\n[7/8] Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Model Comparison - RMSE
    ax1 = plt.subplot(2, 3, 1)
    models = results_df['model']
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, results_df['val_rmse'], width, label='Validation', alpha=0.8)
    ax1.bar(x + width/2, results_df['test_rmse'], width, label='Test', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=10)
    ax1.set_ylabel('RMSE', fontsize=10)
    ax1.set_title('Model Comparison - RMSE (Lower is Better)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Model Comparison - MAE
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(x - width/2, results_df['val_mae'], width, label='Validation', alpha=0.8)
    ax2.bar(x + width/2, results_df['test_mae'], width, label='Test', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=10)
    ax2.set_ylabel('MAE', fontsize=10)
    ax2.set_title('Model Comparison - MAE (Lower is Better)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Improvement over Baseline
    ax3 = plt.subplot(2, 3, 3)
    baseline_rmse = results_df[results_df['model'] == 'Global Mean']['test_rmse'].values[0]
    improvements = ((baseline_rmse - results_df['test_rmse']) / baseline_rmse * 100)
    colors = ['red' if x < 0 else 'green' for x in improvements]
    ax3.barh(models, improvements, color=colors, alpha=0.7)
    ax3.set_xlabel('% Improvement over Baseline', fontsize=10)
    ax3.set_title('RMSE Improvement over Global Mean', fontsize=11, fontweight='bold')
    ax3.axvline(0, color='black', linestyle='--', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Cold-Start Analysis (if available)
    if cold_start_results:
        ax4 = plt.subplot(2, 3, 4)
        best_model = results_df.loc[results_df['test_rmse'].idxmin(), 'model']
        
        if best_model in cold_start_results:
            cold_data = cold_start_results[best_model]
            categories = list(cold_data.keys())
            rmse_values = [cold_data[cat]['rmse'] for cat in categories]
            n_test = [cold_data[cat]['n_test'] for cat in categories]
            
            bars = ax4.bar(categories, rmse_values, alpha=0.7, color='skyblue')
            ax4.set_ylabel('RMSE', fontsize=10)
            ax4.set_title(f'Cold-Start Analysis - {best_model}', fontsize=11, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add sample counts on bars
            for bar, n in zip(bars, n_test):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'n={n}', ha='center', va='bottom', fontsize=8)
    
    # 5. Prediction Distribution
    ax5 = plt.subplot(2, 3, 5)
    best_model_idx = results_df['test_rmse'].idxmin()
    best_predictions = results_df.loc[best_model_idx, 'predictions']
    
    ax5.hist(best_predictions, bins=30, alpha=0.7, label='Predictions', color='orange', edgecolor='black')
    ax5.axvline(np.mean(best_predictions), color='red', linestyle='--', 
                label=f'Mean={np.mean(best_predictions):.2f}')
    ax5.set_xlabel('Predicted Rating', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title(f'Prediction Distribution - {results_df.loc[best_model_idx, "model"]}', 
                  fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['model'],
            f"{row['test_rmse']:.4f}",
            f"{row['test_mae']:.4f}",
            f"{((baseline_rmse - row['test_rmse']) / baseline_rmse * 100):.1f}%"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Model', 'Test RMSE', 'Test MAE', 'vs Baseline'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Highlight best model
    best_idx = results_df['test_rmse'].idxmin() + 1  # +1 for header
    for j in range(4):
        table[(best_idx, j)].set_facecolor('#90EE90')
    
    ax6.set_title('Performance Summary', fontsize=11, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("  Saved visualization: model_evaluation_results.png")


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    # Load data
    train, val, test, recipes = load_and_preprocess_data()
    recipes_feat = extract_recipe_features(recipes)
    
    # Initialize models
    print("\n[3/8] Initializing models...")
    models = {
        'Global Mean': GlobalMeanBaseline(),
        'Bias Model': BiasBaseline(alpha=5),
        'SVD (50)': MatrixFactorizationSVD(n_factors=50, n_epochs=20, lr=0.005, reg=0.02),
        'SVD (100)': MatrixFactorizationSVD(n_factors=100, n_epochs=15, lr=0.005, reg=0.02),
        'Content-Based': ContentBasedRecommender(recipes_feat),
    }
    
    # Train models
    print("\n[4/8] Training models...")
    results = []
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        try:
            model.fit(train)
            result = evaluate_model(model, train, val, test, name)
            results.append(result)
        except Exception as e:
            print(f"    Error training {name}: {str(e)}")
            continue
    
    # Train Hybrid model (SVD + Content-Based)
    print("\n  Training Hybrid Model...")
    try:
        hybrid_cf = MatrixFactorizationSVD(n_factors=50, n_epochs=20, lr=0.005, reg=0.02)
        hybrid_cb = ContentBasedRecommender(recipes_feat)
        hybrid_model = HybridRecommender(hybrid_cf, hybrid_cb, alpha=0.75)
        hybrid_model.fit(train)
        result = evaluate_model(hybrid_model, train, val, test, 'Hybrid (SVD+CB)')
        results.append(result)
    except Exception as e:
        print(f"    Error training Hybrid: {str(e)}")
    
    # Create results dataframe
    print("\n[5/8] Compiling results...")
    results_df = pd.DataFrame([
        {
            'model': r['model'],
            'val_rmse': r['val_rmse'],
            'val_mae': r['val_mae'],
            'test_rmse': r['test_rmse'],
            'test_mae': r['test_mae'],
            'predictions': r['predictions']
        }
        for r in results
    ])
    
    # Sort by test RMSE
    results_df = results_df.sort_values('test_rmse')
    
    # Cold-start analysis
    print("\n[6/8] Running cold-start analysis...")
    cold_start_results = {}
    
    for name, model in models.items():
        if name in ['Global Mean']:  # Skip trivial baselines
            continue
        try:
            cs_result = cold_start_analysis(model, train, test, name)
            cold_start_results[name] = cs_result
        except Exception as e:
            print(f"    Error in cold-start for {name}: {str(e)}")
    
    # Visualize results
    plot_results(results_df, cold_start_results)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print("\nModel Performance (sorted by Test RMSE):")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['model']:20s} | Test RMSE: {row['test_rmse']:.4f} | "
              f"Test MAE: {row['test_mae']:.4f}")
    
    best_model = results_df.iloc[0]
    baseline_rmse = results_df[results_df['model'] == 'Global Mean']['test_rmse'].values[0]
    improvement = (baseline_rmse - best_model['test_rmse']) / baseline_rmse * 100
    
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model['model']}")
    print(f"Test RMSE: {best_model['test_rmse']:.4f}")
    print(f"Test MAE: {best_model['test_mae']:.4f}")
    print(f"Improvement over baseline: {improvement:.2f}%")
    print("="*80)
    
    # Save results
    results_df[['model', 'val_rmse', 'val_mae', 'test_rmse', 'test_mae']].to_csv(
        'model_results.csv', index=False
    )
    print("\n[8/8] Results saved to 'model_results.csv'")
    print("\nDone! Check 'model_evaluation_results.png' for visualizations.")


if __name__ == "__main__":
    main()

