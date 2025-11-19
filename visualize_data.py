"""
Data Visualization Script for Food.com Recipe Dataset
Creates plots to understand data characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import ast

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("Loading data...")
raw_recipes = pd.read_csv('RAW_recipes.csv')
raw_interactions = pd.read_csv('RAW_interactions.csv')
interactions_train = pd.read_csv('interactions_train.csv')

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Rating Distribution
ax1 = plt.subplot(3, 3, 1)
rating_counts = raw_interactions['rating'].value_counts().sort_index()
ax1.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
ax1.set_xlabel('Rating', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.set_title('Rating Distribution (Highly Skewed!)', fontsize=11, fontweight='bold')
ax1.set_xticks([0, 1, 2, 3, 4, 5])
for i, v in enumerate(rating_counts.values):
    ax1.text(rating_counts.index[i], v, f'{v:,}', ha='center', va='bottom', fontsize=8)

# 2. User Activity Distribution (log scale)
ax2 = plt.subplot(3, 3, 2)
user_activity = raw_interactions.groupby('user_id').size()
ax2.hist(user_activity, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Interactions per User', fontsize=10)
ax2.set_ylabel('Number of Users (log scale)', fontsize=10)
ax2.set_yscale('log')
ax2.set_title('User Activity Distribution (Long Tail!)', fontsize=11, fontweight='bold')
ax2.axvline(user_activity.median(), color='red', linestyle='--', label=f'Median: {user_activity.median():.0f}')
ax2.legend(fontsize=8)

# 3. Recipe Popularity Distribution (log scale)
ax3 = plt.subplot(3, 3, 3)
recipe_popularity = raw_interactions.groupby('recipe_id').size()
ax3.hist(recipe_popularity, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Interactions per Recipe', fontsize=10)
ax3.set_ylabel('Number of Recipes (log scale)', fontsize=10)
ax3.set_yscale('log')
ax3.set_title('Recipe Popularity Distribution (Long Tail!)', fontsize=11, fontweight='bold')
ax3.axvline(recipe_popularity.median(), color='red', linestyle='--', label=f'Median: {recipe_popularity.median():.0f}')
ax3.legend(fontsize=8)

# 4. Cooking Time Distribution
ax4 = plt.subplot(3, 3, 4)
cooking_time = raw_recipes['minutes'][raw_recipes['minutes'] <= 180]  # Filter outliers
ax4.hist(cooking_time, bins=50, color='gold', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Cooking Time (minutes, filtered ≤180)', fontsize=10)
ax4.set_ylabel('Number of Recipes', fontsize=10)
ax4.set_title('Cooking Time Distribution', fontsize=11, fontweight='bold')
ax4.axvline(raw_recipes['minutes'].median(), color='red', linestyle='--', 
            label=f'Median: {raw_recipes["minutes"].median():.0f} min')
ax4.legend(fontsize=8)

# 5. Number of Ingredients Distribution
ax5 = plt.subplot(3, 3, 5)
ax5.hist(raw_recipes['n_ingredients'], bins=30, color='plum', edgecolor='black', alpha=0.7)
ax5.set_xlabel('Number of Ingredients', fontsize=10)
ax5.set_ylabel('Number of Recipes', fontsize=10)
ax5.set_title('Recipe Complexity (Ingredients)', fontsize=11, fontweight='bold')
ax5.axvline(raw_recipes['n_ingredients'].median(), color='red', linestyle='--',
            label=f'Median: {raw_recipes["n_ingredients"].median():.0f}')
ax5.legend(fontsize=8)

# 6. Number of Steps Distribution
ax6 = plt.subplot(3, 3, 6)
ax6.hist(raw_recipes['n_steps'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax6.set_xlabel('Number of Steps', fontsize=10)
ax6.set_ylabel('Number of Recipes', fontsize=10)
ax6.set_title('Recipe Complexity (Steps)', fontsize=11, fontweight='bold')
ax6.axvline(raw_recipes['n_steps'].median(), color='red', linestyle='--',
            label=f'Median: {raw_recipes["n_steps"].median():.0f}')
ax6.legend(fontsize=8)

# 7. Temporal Pattern - Interactions over Years
ax7 = plt.subplot(3, 3, 7)
raw_interactions['year'] = pd.to_datetime(raw_interactions['date']).dt.year
interactions_by_year = raw_interactions.groupby('year').size()
ax7.plot(interactions_by_year.index, interactions_by_year.values, marker='o', 
         linewidth=2, markersize=6, color='darkblue')
ax7.fill_between(interactions_by_year.index, interactions_by_year.values, alpha=0.3, color='skyblue')
ax7.set_xlabel('Year', fontsize=10)
ax7.set_ylabel('Number of Interactions', fontsize=10)
ax7.set_title('Temporal Pattern (2000-2018)', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Nutrition - Calories Distribution
ax8 = plt.subplot(3, 3, 8)
nutrition_sample = raw_recipes['nutrition'].head(10000).apply(ast.literal_eval)
calories = [n[0] for n in nutrition_sample if n[0] <= 1000]  # Filter outliers
ax8.hist(calories, bins=50, color='orange', edgecolor='black', alpha=0.7)
ax8.set_xlabel('Calories (filtered ≤1000)', fontsize=10)
ax8.set_ylabel('Number of Recipes', fontsize=10)
ax8.set_title('Calorie Distribution', fontsize=11, fontweight='bold')
ax8.axvline(np.median(calories), color='red', linestyle='--',
            label=f'Median: {np.median(calories):.0f}')
ax8.legend(fontsize=8)

# 9. Top Recipe Tags
ax9 = plt.subplot(3, 3, 9)
all_tags = []
for tags in raw_recipes['tags'].head(20000):
    try:
        tag_list = ast.literal_eval(tags)
        all_tags.extend(tag_list)
    except:
        pass
tag_counts = Counter(all_tags)
top_tags = dict(tag_counts.most_common(15))
ax9.barh(list(top_tags.keys())[::-1], list(top_tags.values())[::-1], color='teal', alpha=0.7)
ax9.set_xlabel('Count', fontsize=10)
ax9.set_title('Top 15 Recipe Tags', fontsize=11, fontweight='bold')
ax9.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig('food_dataset_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved as 'food_dataset_analysis.png'")

# Print key statistics
print("\n" + "="*80)
print("KEY STATISTICS SUMMARY")
print("="*80)
print(f"Total Recipes: {len(raw_recipes):,}")
print(f"Total Users: {raw_interactions['user_id'].nunique():,}")
print(f"Total Interactions: {len(raw_interactions):,}")
print(f"Date Range: {raw_interactions['date'].min()} to {raw_interactions['date'].max()}")
print(f"\nMedian User Activity: {user_activity.median():.0f} interactions")
print(f"Median Recipe Popularity: {recipe_popularity.median():.0f} interactions")
print(f"Median Cooking Time: {raw_recipes['minutes'].median():.0f} minutes")
print(f"Median Recipe Complexity: {raw_recipes['n_steps'].median():.0f} steps")
print(f"Median Ingredients: {raw_recipes['n_ingredients'].median():.0f} ingredients")
print(f"\nRating Distribution:")
print(f"   5 stars: {(raw_interactions['rating']==5).sum()/len(raw_interactions)*100:.1f}%")
print(f"   4 stars: {(raw_interactions['rating']==4).sum()/len(raw_interactions)*100:.1f}%")
print(f"   3 stars: {(raw_interactions['rating']==3).sum()/len(raw_interactions)*100:.1f}%")
print(f"   <=2 stars: {(raw_interactions['rating']<=2).sum()/len(raw_interactions)*100:.1f}%")
print(f"\n[WARNING] Data Sparsity: 99.9978% (major challenge!)")
print("="*80)

plt.show()

