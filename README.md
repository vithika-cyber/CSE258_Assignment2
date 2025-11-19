# CSE 258 Assignment 2 - Recipe Recommendation System

## 📋 Project Files

This directory contains everything you need to get started with your recipe recommendation project.

### 📊 Data Files (Already Present)
- `RAW_recipes.csv` - 231,637 recipes with full metadata
- `RAW_interactions.csv` - 1,132,367 user-recipe interactions
- `interactions_train.csv` - Training set (698,901 interactions)
- `interactions_validation.csv` - Validation set (7,023 interactions)  
- `interactions_test.csv` - Test set (12,455 interactions)
- `PP_recipes.csv` - Preprocessed recipes with tokenized features
- `PP_users.csv` - Preprocessed user profiles
- `ingr_map.pkl` - Ingredient ID mapping

### 📖 Documentation (Just Created for You)

1. **`GETTING_STARTED.md`** ⭐ **START HERE!**
   - Quick start guide
   - Step-by-step instructions
   - Implementation roadmap
   - Assignment report structure

2. **`DATASET_SUMMARY_AND_RECOMMENDATIONS.md`**
   - Comprehensive dataset analysis
   - Detailed statistics and characteristics
   - Recommendation task options
   - Challenge analysis (sparsity, cold-start, bias)
   - Suggested models and approaches

3. **`starter_template.py`**
   - Ready-to-run Python template
   - 4 baseline models already implemented:
     - Global Mean Baseline
     - User Mean Baseline
     - Recipe Mean Baseline
     - User + Recipe Bias Model
   - Feature engineering functions
   - Evaluation metrics
   - TODOs for advanced models

4. **`visualize_data.py`**
   - Data visualization script
   - Creates 9-panel analysis plot
   - Prints key statistics
   - Run this first to understand the data!

---

## 🚀 Quick Start (3 Steps)

### Step 1: Visualize the Data
```bash
python visualize_data.py
```
This creates `food_dataset_analysis.png` showing rating distribution, user activity, recipe features, etc.

### Step 2: Read the Documentation
Start with **`GETTING_STARTED.md`** for a complete walkthrough.

### Step 3: Run the Baseline Template
```bash
python starter_template.py
```
This will train and evaluate 4 baseline models on your data.

---

## 🎯 Recommended Project

**Task**: Personalized Recipe Rating Prediction & Top-K Recommendation

**Why?**
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

---

## 📈 Models to Implement

### Baselines (Already Implemented ✅)
1. Global mean prediction
2. User mean prediction
3. Recipe mean prediction
4. User + Recipe bias model

### Advanced Models (TODO)
1. **Matrix Factorization** (SVD, ALS)
2. **Content-Based Filtering** (using recipe features)
3. **Hybrid Models** (combine CF + content)
4. **Neural Collaborative Filtering** (optional)

---

## 📏 Evaluation Metrics

**Rating Prediction**:
- RMSE (Root Mean Squared Error) - primary metric
- MAE (Mean Absolute Error)

**Top-K Recommendation**:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)

**Cold-Start**:
- Performance on users with 1-3 interactions vs 10+ interactions

---

## 📝 For Your Report

Your assignment requires 4 sections:

1. **Predictive Task** - Task definition, evaluation strategy, baselines
2. **Exploratory Analysis** - Data context, preprocessing, statistics, plots
3. **Modeling** - ML formulation, model comparison, implementation details
4. **Evaluation** - Results, baseline comparison, metrics justification

See **`GETTING_STARTED.md`** Section 6 for detailed guidance.

---

## 🛠️ Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install scikit-surprise  # For collaborative filtering
```

---

## 📚 Files Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `GETTING_STARTED.md` | Complete walkthrough | Read FIRST |
| `DATASET_SUMMARY_AND_RECOMMENDATIONS.md` | Deep data analysis | For EDA section |
| `visualize_data.py` | Create visualizations | Run at start |
| `starter_template.py` | Code template | Implementation |
| This `README.md` | Quick reference | Anytime |

---

## ✅ Next Steps

1. ✅ Run `python visualize_data.py`
2. ✅ Read `GETTING_STARTED.md`
3. ✅ Read `DATASET_SUMMARY_AND_RECOMMENDATIONS.md`  
4. ✅ Run `python starter_template.py`
5. ✅ Start implementing advanced models
6. ✅ Create your analysis notebook/report

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

## 🎯 Success Criteria

- ✅ Beat baseline models (RMSE < 0.85)
- ✅ Implement at least 2 advanced models
- ✅ Address cold-start problem
- ✅ Use content features effectively
- ✅ Comprehensive evaluation and analysis

---

## 🎉 You're All Set!

Everything is prepared for you to start working on this interesting recommendation problem. The data is fascinating, the task is well-defined, and you have all the tools you need.

**Good luck and have fun! 🚀**

---

*Generated for CSE 258 Assignment 2 - Food.com Recipe Dataset*

