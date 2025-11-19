# CSE 258 Assignment 2 - Recipe Recommendation System

## 📋 Project Overview

This repository contains code and analysis for a recipe recommendation system using the Food.com dataset.

---

## 📊 Data Files (NOT INCLUDED - Download Required)

**⚠️ IMPORTANT**: Dataset files are NOT included in this repository due to their large size (800+ MB total).

**Required Data Files**:
- `RAW_recipes.csv` - 231,637 recipes with full metadata (~280 MB)
- `RAW_interactions.csv` - 1,132,367 user-recipe interactions (~332 MB)
- `interactions_train.csv` - Training set (698,901 interactions)
- `interactions_validation.csv` - Validation set (7,023 interactions)  
- `interactions_test.csv` - Test set (12,455 interactions)
- `PP_recipes.csv` - Preprocessed recipes with tokenized features (~195 MB)
- `PP_users.csv` - Preprocessed user profiles
- `ingr_map.pkl` - Ingredient ID mapping

**📥 Where to Download**:
- **Original Dataset**: [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
  - Download from Kaggle (requires free Kaggle account)

**📂 Setup Instructions**:
```bash
# 1. Download all required data files
# 2. Place them in the project root directory (same folder as this README)
# 3. Your directory structure should look like:
#    CSE 258_A2/
#    ├── RAW_recipes.csv
#    ├── RAW_interactions.csv
#    ├── interactions_train.csv
#    ├── interactions_validation.csv
#    ├── interactions_test.csv
#    ├── PP_recipes.csv
#    ├── PP_users.csv
#    ├── ingr_map.pkl
#    ├── starter_template.py
#    ├── advanced_models.py
#    └── README.md
```

---

## 🚀 Quick Start

### Step 1: Download the Data ⚠️
**Before running any code, download all required dataset files** (see "Data Files" section above).
Place them in the project root directory.

### Step 2: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install scikit-surprise  # For collaborative filtering
```

### Step 3: Visualize the Data (Optional)
```bash
python visualize_data.py
```
This creates `food_dataset_analysis.png` showing rating distribution, user activity, recipe features, etc.

### Step 4: Run the Models
```bash
# Run baseline models
python starter_template.py

# Run advanced models
python advanced_models.py
```

---

## 📁 Repository Files

### Python Scripts
- **`starter_template.py`** - Baseline models implementation
  - Global mean prediction
  - User mean prediction
  - Recipe mean prediction
  - User + Recipe bias model
  
- **`advanced_models.py`** - Advanced model implementations
  - Matrix Factorization (SVD, ALS)
  - Content-based filtering
  - Hybrid models
  
- **`visualize_data.py`** - Data visualization and analysis script

### Documentation & Analysis
- **`DATASET_SUMMARY_AND_RECOMMENDATIONS.md`** - Comprehensive dataset analysis and recommendations
- **`RESULTS_ANALYSIS.md`** - Model evaluation results and insights
- **`model_results.csv`** - Numerical results from model experiments
- **`food_dataset_analysis.png`** - Visual analysis of the dataset
- **`model_evaluation_results.png`** - Model performance visualizations

### Assignment Materials
- **`158 _ 258 2025 Assignment 2.pdf`** - Assignment instructions

---

## 🎯 Project Task

**Task**: Personalized Recipe Rating Prediction & Top-K Recommendation

**Why This Task?**
- ✅ Perfect for CSE 258 (collaborative filtering, matrix factorization)
- ✅ Real-world application
- ✅ Multiple interesting challenges
- ✅ Clear evaluation metrics

**Components**:
1. **Rating Prediction** (Regression) - Predict rating (0-5) for user-recipe pairs
2. **Top-K Recommendation** (Ranking) - Recommend K best recipes per user
3. **Cold-Start Analysis** - Handle sparse users/recipes

---

## 📊 Dataset Highlights

- **231,637** unique recipes
- **226,570** unique users  
- **1,132,367** interactions over 18 years (2000-2018)
- **99.99%** sparsity (major challenge!)
- **72%** of ratings are 5-stars (highly skewed)
- **Median user** has only 1 interaction (cold-start problem)

**Rich Features Available**:
- Recipe tags (cuisine, dietary, occasion)
- Nutrition info (calories, protein, fat, etc.)
- Cooking time, ingredients, steps
- User reviews and ratings

For detailed statistics and analysis, see `DATASET_SUMMARY_AND_RECOMMENDATIONS.md`.

---

## 📈 Models Implemented

### Baselines (in `starter_template.py`)
1. ✅ Global mean prediction
2. ✅ User mean prediction
3. ✅ Recipe mean prediction
4. ✅ User + Recipe bias model

### Advanced Models (in `advanced_models.py`)
1. ✅ Matrix Factorization (SVD, ALS)
2. ✅ Content-Based Filtering (using recipe features)
3. ✅ Hybrid Models (combine CF + content)

For detailed results and analysis, see `RESULTS_ANALYSIS.md`.

---

## 📏 Evaluation Metrics

**Rating Prediction**:
- RMSE (Root Mean Squared Error) - primary metric
- MAE (Mean Absolute Error)

**Top-K Recommendation**:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)

**Cold-Start Analysis**:
- Performance on users with 1-3 interactions vs 10+ interactions

---

## 📝 Assignment Report Structure

Your assignment requires 4 sections:

1. **Predictive Task** - Task definition, evaluation strategy, baselines
2. **Exploratory Analysis** - Data context, preprocessing, statistics, plots
3. **Modeling** - ML formulation, model comparison, implementation details
4. **Evaluation** - Results, baseline comparison, metrics justification

Refer to:
- `DATASET_SUMMARY_AND_RECOMMENDATIONS.md` for exploratory analysis
- `RESULTS_ANALYSIS.md` for evaluation insights
- `model_results.csv` for numerical results
- `food_dataset_analysis.png` and `model_evaluation_results.png` for visualizations

---

## 💡 Key Insights

**Main Challenge**: Extreme sparsity (99.99%)
- Most users have 1-2 interactions only
- Cold-start is a critical problem
- Need to leverage content features

**Data Bias**: 72% ratings are 5-stars
- Consider binary classification (liked/not liked)
- Focus on ranking rather than exact rating prediction

**Rich Features**: Tags, ingredients, nutrition
- Perfect for hybrid models
- Can help with cold-start problem
- Enable interesting constraints (dietary, time, calories)

---

## ✅ Workflow

1. ✅ Download required dataset files (see "Data Files" section)
2. ✅ Install dependencies (`pip install pandas numpy scikit-learn matplotlib seaborn scikit-surprise`)
3. ✅ Run `python visualize_data.py` to understand the data
4. ✅ Read `DATASET_SUMMARY_AND_RECOMMENDATIONS.md` for detailed analysis
5. ✅ Run `python starter_template.py` for baseline models
6. ✅ Run `python advanced_models.py` for advanced models
7. ✅ Review `RESULTS_ANALYSIS.md` for insights
8. ✅ Use results to write your assignment report

---

## 🎯 Success Criteria

- ✅ Beat baseline models (RMSE < 0.85)
- ✅ Implement at least 2 advanced models
- ✅ Address cold-start problem
- ✅ Use content features effectively
- ✅ Comprehensive evaluation and analysis

---

## 🎉 Ready to Go!

Everything is prepared for you to start working on this recommendation system project. Download the data, run the code, and analyze the results!

**Good luck! 🚀**

---

*CSE 258 Assignment 2 - Food.com Recipe Dataset*
