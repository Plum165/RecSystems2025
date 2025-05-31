# 🛒 Product Recommendation System using LightGBM

**Author**: Moegamat Samsodien  
**Date**: 31/05/2025

This project is a recommendation system that predicts which products a customer is likely to purchase next, based on historical e-commerce transaction data. It uses a **LightGBM** binary classification model to rank potential product recommendations for each user.

---

## 🚀 Features

- Loads and preprocesses a dataset of retail transactions
- Engineers features related to user purchase behavior (recency, frequency, monetary)
- Encodes user and item data for machine learning
- Generates **negative samples** for implicit feedback learning
- Trains a **LightGBM** model on user-item interactions
- Recommends **top N unseen items** for any given user
- Displays both the **StockCode** and **item description** for each recommendation

---

## 📁 Dataset

The dataset used is assumed to follow the format of the **UCI Online Retail Dataset**, and must contain the following columns:

- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

> 📌 **Note:** Ensure the dataset is saved as `data.csv` in the same directory.

---

## 🧠 Model

The model uses:

- **Label Encoding** for users and products
- **Negative Sampling** to balance implicit feedback
- **LightGBM (Gradient Boosting)** for classification

The label is binary:
- `1` = User purchased the item
- `0` = Negative sampled (user did not purchase this item)

---

## 🛠️ How to Run

1. Make sure you have Python 3.7+ installed
2. Install required packages:
   ```bash
   pip install pandas numpy lightgbm scikit-learn
3. Download the dataset from https://www.kaggle.com/datasets/carrie1/ecommerce-data
   It was too large to upload on GitHub
## 📚 Learning Reflection
This project was built as part of the FNB DataQuest 2025 challenge. It was my first real experience with recommendation systems, machine learning pipelines, and working with large-scale retail data.

Despite limited time and exam season, I’m proud of what I achieved and excited to explore more advanced techniques (e.g., matrix factorization, neural collaborative filtering) in future iterations.

## NOTE
The training model for R won't work as I have not included the dataset for it.
