# Food.com Recipe Dataset - Analysis & Recommendation Tasks

## 📊 Dataset Overview

### Source & Context
- **Origin**: Food.com (formerly GeniusKitchen)
- **Time Span**: 18 years (2000-2018)
- **Size**: 
  - 231,637 unique recipes
  - 226,570 unique users
  - 1,132,367 user-recipe interactions
- **Research Paper**: "Generating Personalized Recipes from Historical User Preferences" (EMNLP 2019)

### Data Files Available
1. **Interaction Splits** (for model training/testing):
   - `interactions_train.csv` (698,901 interactions)
   - `interactions_validation.csv` (7,023 interactions)
   - `interactions_test.csv` (12,455 interactions)

2. **Raw Data**:
   - `RAW_recipes.csv` - Complete recipe information
   - `RAW_interactions.csv` - All user-recipe interactions with reviews

3. **Preprocessed Data** (for reproduction):
   - `PP_recipes.csv` - Tokenized recipe features
   - `PP_users.csv` - User feature aggregations

---

## 🔍 Key Data Characteristics

### 1. Rating Distribution
- **Highly Skewed**: 72.1% of ratings are 5-stars
- **Mean Rating**: 4.41 / 5.0
- **Rating Breakdown**:
  - 5 stars: 816,364 (72.1%)
  - 4 stars: 187,360 (16.5%)
  - 3 stars: 40,855 (3.6%)
  - 2 stars: 14,123 (1.2%)
  - 1 star: 12,818 (1.1%)
  - 0 stars: 60,847 (5.4%) ⚠️ *likely missing/no rating*

### 2. Sparsity Issues
- **Matrix Size**: 226,570 users × 231,637 recipes
- **Sparsity**: 99.9978% (extremely sparse!)
- **Challenge**: Most users have rated very few recipes
  - Median: 1 interaction per user
  - 75th percentile: 2 interactions per user

### 3. User & Recipe Statistics
**Users**:
- Mean interactions per user: 5.0
- Median: 1.0 (very sparse!)
- Max (power user): 7,671 interactions

**Recipes**:
- Mean interactions per recipe: 4.9
- Median: 2.0
- Most popular recipe: 1,613 interactions

### 4. Recipe Features
- **Cooking Time**: Median 40 minutes (mean inflated by outliers)
- **Complexity**: ~9-10 steps on average
- **Ingredients**: ~9 ingredients on average
- **Tags**: Rich categorization (cuisine, dietary, occasion, etc.)
- **Nutrition**: 7 features (calories, fat, sugar, sodium, protein, saturated fat, carbs)

### 5. Temporal Patterns
- Peak activity: 2008-2009 (~160K interactions/year)
- Decline after 2009
- Data from 2000-2018

---

## 🎯 Recommended Recommendation Tasks

### **Task 1: Rating Prediction (Regression)**

**Objective**: Predict the rating a user would give to a recipe

**Why This Task?**
- ✅ Classic recommender systems problem (covered in CSE 258)
- ✅ Clear evaluation metrics
- ✅ Multiple baseline models to compare
- ✅ Can incorporate both collaborative and content-based features

**Evaluation Metrics**:
- RMSE (Root Mean Squared Error) - primary metric
- MAE (Mean Absolute Error)
- Accuracy for binary classification (liked/not liked with threshold)

**Baselines**:
1. **Trivial**: Predict global mean rating (4.41)
2. **User-based**: Predict user's mean rating
3. **Recipe-based**: Predict recipe's mean rating
4. **User + Recipe bias**: Combine both biases

**Models to Implement**:
1. **Latent Factor Models** (Matrix Factorization):
   - Basic SVD
   - SVD++ (with implicit feedback)
   - NMF (Non-negative Matrix Factorization)
   
2. **Content-Based**:
   - Recipe similarity using tags, ingredients, nutrition
   - User profile based on past preferences
   
3. **Hybrid Models**:
   - Factorization Machines (combine CF + content features)
   - Neural Collaborative Filtering

**Features to Use**:
- User ID, Recipe ID (collaborative filtering)
- Recipe tags (cuisine, dietary, time, etc.)
- Nutrition information
- Number of ingredients/steps (complexity)
- Recipe popularity
- User activity level

---

### **Task 2: Top-K Recipe Recommendation (Ranking)**

**Objective**: Recommend K recipes most likely to be highly rated by a user

**Why This Task?**
- ✅ More practical than rating prediction
- ✅ Handles implicit feedback (views vs ratings)
- ✅ Can test personalization quality

**Evaluation Metrics**:
- **Precision@K**: Of K recommended recipes, how many were actually liked?
- **Recall@K**: Of all recipes user liked, how many were in top K?
- **NDCG@K**: Normalized Discounted Cumulative Gain (position-aware)
- **Hit Rate@K**: Did we get at least one correct recommendation?

**Baselines**:
1. **Popularity**: Recommend most popular recipes
2. **Random**: Random recommendations
3. **User Average**: Recommend recipes similar to user's high-rated ones
4. **Item-Item CF**: Collaborative filtering based on recipe similarity

**Models to Implement**:
1. **Collaborative Filtering**:
   - Item-based CF (recipe similarity)
   - User-based CF (user similarity)
   - Matrix Factorization (get latent factors, rank by predicted rating)

2. **Learning to Rank**:
   - BPR (Bayesian Personalized Ranking)
   - WARP (Weighted Approximate-Rank Pairwise)

3. **Content-Based**:
   - TF-IDF on recipe features
   - Recipe similarity using tags/ingredients

---

### **Task 3: Cold-Start Recommendation (Hybrid)**

**Objective**: Recommend recipes for new users or new recipes with few interactions

**Why This Task?**
- ✅ Addresses a real-world challenge
- ✅ Forces use of content features
- ✅ Tests model robustness
- ✅ Given sparsity (median 1 interaction), this is critical!

**Evaluation**:
- Split data into "cold" users (1-2 interactions) vs "warm" users
- Measure performance degradation
- Test on recipes with few ratings

**Models**:
1. **Pure Content-Based**: Only use recipe features
2. **Hybrid Models**: Combine CF when available with content features
3. **Transfer Learning**: Use features from similar users/recipes

---

## 💡 My Recommendation: **Combined Task**

I suggest combining Tasks 1 & 2 for a comprehensive project:

### **"Personalized Recipe Recommendation with Rating Prediction"**

**Part A: Rating Prediction**
- Predict ratings for validation/test set
- Compare multiple baseline models
- Report RMSE, MAE

**Part B: Top-10 Recommendation**
- Use predicted ratings to rank recipes
- Recommend top 10 recipes per user
- Report Precision@10, Recall@10, NDCG@10

**Part C: Cold-Start Analysis**
- Analyze performance on sparse users (1-3 interactions)
- Compare CF vs Hybrid approaches

---

## 📋 Project Structure for Assignment

### 1. Exploratory Data Analysis
- Rating distribution (show the skew)
- Sparsity analysis
- User/recipe statistics
- Feature distributions (cooking time, ingredients, nutrition)
- Temporal patterns
- Tag analysis

### 2. Data Preprocessing
- Handle 0-ratings (treat as missing?)
- Feature engineering:
  - Extract nutrition features
  - Parse tags into categories
  - Create user/recipe popularity features
  - Time-based features
- Train/validation/test split (already provided!)
- Normalization/scaling

### 3. Baseline Models
1. Global average
2. User bias model
3. Recipe bias model
4. User + Recipe bias model
5. Popularity-based recommendation

### 4. Advanced Models
1. **Matrix Factorization** (SVD, ALS)
2. **Content-Based Filtering** (using tags, ingredients, nutrition)
3. **Hybrid Model** (Factorization Machines or similar)
4. **Optional**: Neural Collaborative Filtering

### 5. Evaluation
- Rating prediction: RMSE, MAE
- Recommendation: Precision@K, Recall@K, NDCG@K
- Cold-start analysis
- Feature importance analysis
- Error analysis

---

## 🚀 Quick Start Ideas

### Interesting Research Questions:
1. **Does recipe complexity affect ratings?** (steps, ingredients)
2. **Can we predict ratings using only content features?** (for cold-start)
3. **How does nutrition information help recommendation?** (dietary preferences)
4. **Are certain tags more predictive of high ratings?** (easy, quick, dietary)
5. **How do temporal patterns affect preferences?** (seasonal recipes)

### Potential Constraints/Extensions:
- **Nutritional constraints**: Recommend low-calorie recipes
- **Time constraints**: Quick recipes (<30 min)
- **Dietary filters**: Vegetarian, vegan, low-carb
- **Diversity**: Recommend diverse cuisines
- **Serendipity**: Balance accuracy with novelty

---

## ⚠️ Key Challenges to Address

1. **Extreme Sparsity** (99.99%)
   - Solution: Hybrid models, regularization, cold-start handling

2. **Rating Bias** (72% are 5-stars)
   - Solution: Consider binary classification (liked/not liked)
   - Solution: Focus on ranking rather than absolute ratings

3. **Long-Tail Distribution**
   - 50% of users have only 1 interaction
   - Solution: Content-based for sparse users

4. **Cold-Start Problem**
   - Many users/recipes with few interactions
   - Solution: Leverage content features (tags, ingredients, nutrition)

5. **Implicit Feedback**
   - 0-ratings might mean "no rating" not "bad"
   - Solution: Treat as missing data or implicit negative

---

## 📊 Suggested Evaluation Protocol

### 1. Rating Prediction
```python
# Use provided train/val/test splits
# Metrics: RMSE, MAE on test set
# Compare against baselines
```

### 2. Top-K Recommendation
```python
# For each user in test set:
#   - Hide their interactions
#   - Rank all recipes
#   - Check if true interactions are in top K
# Metrics: Precision@K, Recall@K, NDCG@K for K=5,10,20
```

### 3. Cold-Start Evaluation
```python
# Filter test users by # of training interactions
# Groups: 1, 2-5, 6-10, 11+ interactions
# Compare performance across groups
```

---

## 🛠️ Tools & Libraries

Recommended Python packages:
- `pandas`, `numpy` - data manipulation
- `scikit-learn` - baseline models, evaluation
- `scipy` - sparse matrices
- `surprise` - recommender system library
- `implicit` - fast implicit CF models
- `matplotlib`, `seaborn` - visualization
- `tensorflow`/`pytorch` - neural models (optional)

---

## 📖 Relevant Course Concepts (CSE 258)

- **Collaborative Filtering**: User-based, Item-based
- **Matrix Factorization**: SVD, ALS, SGD
- **Regularization**: Prevent overfitting in sparse data
- **Bias Models**: User/item biases
- **Content-Based Filtering**: Feature-based similarity
- **Hybrid Models**: Combining CF and content
- **Evaluation**: RMSE, MAE, Precision/Recall, NDCG
- **Cold-Start Problem**: Handling new users/items

---

## 🎓 Summary & Next Steps

**Dataset Strengths**:
✅ Large scale (230K recipes, 1M+ interactions)
✅ Rich features (tags, ingredients, nutrition, reviews)
✅ Real-world application (food recommendation)
✅ Provided train/val/test splits
✅ 18 years of temporal data

**Dataset Challenges**:
⚠️ Extreme sparsity (99.99%)
⚠️ Rating bias (72% are 5-stars)
⚠️ Long-tail distribution (median 1 interaction per user)

**Recommended Approach**:
1. Start with rating prediction (easier to debug)
2. Implement solid baselines first
3. Add complexity gradually (CF → Content → Hybrid)
4. Focus on cold-start problem (most relevant given sparsity)
5. Consider binary classification or ranking instead of regression

**Success Criteria**:
- Beat baseline models significantly
- Demonstrate value of content features
- Handle cold-start users effectively
- Show interpretable results (which features matter?)

Good luck with your project! 🚀

