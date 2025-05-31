#Moegamat Samsodien
#31/05/2025
#Program that uses a model to predict items (stockCode) from a dataset.
import pandas as pd
import numpy as np
import random
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. Load and preprocess stock data
# ----------------------------
def load_data():
    df = pd.read_csv('data.csv', encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

def preprocess_data(df):
    max_date = df['InvoiceDate'].max()
    recency_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).reset_index().rename(columns={
        'InvoiceDate': 'recency_days',
        'InvoiceNo': 'frequency',
        'Quantity': 'total_quantity',
        'TotalPrice': 'total_spent'
    })

    item_freq = df.groupby(['CustomerID', 'StockCode']).agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).reset_index().rename(columns={
        'Quantity': 'item_quantity',
        'TotalPrice': 'item_total_spent'
    })

    data = pd.merge(item_freq, recency_df, on='CustomerID', how='left')
    return data

# ----------------------------
# 2. Feature Engineering
# ----------------------------
def encode_features(df):
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    df['user'] = le_user.fit_transform(df['CustomerID'])
    df['item'] = le_item.fit_transform(df['StockCode'])
    return df, le_user, le_item

# ----------------------------
# 3. Generate Negative Samples
# ----------------------------
def generate_negative_samples(df, all_items, num_negatives=1):
    users = df['user'].unique()
    negative_samples = []

    for user in users:
        user_items = set(df[df['user'] == user]['item'])
        for pos_item in user_items:
            for _ in range(num_negatives):
                neg_item = random.choice(all_items)
                while neg_item in user_items:
                    neg_item = random.choice(all_items)
                negative_samples.append([user, neg_item, 0])

    pos_samples = df[['user', 'item']].copy()
    pos_samples['label'] = 1

    neg_samples_df = pd.DataFrame(negative_samples, columns=['user', 'item', 'label'])
    final_df = pd.concat([pos_samples, neg_samples_df], ignore_index=True)
    return final_df

# ----------------------------
# 4. Train LightGBM Model
# ----------------------------
def train_model(train_df):
    X = train_df[['user', 'item']]
    y = train_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    return model

# ----------------------------
# 5. Recommend Items
# ----------------------------
def recommend_items(model, user_id, le_user, le_item, all_items, df, top_k=10):
    user_id = str(user_id).strip()
    if user_id not in le_user.classes_:
        print("User not found in training data.")
        return []

    user_enc = le_user.transform([user_id])[0]
    user_items = set(df[df['CustomerID'] == user_id]['StockCode'])
    user_item_enc = set(le_item.transform(list(user_items)))

    candidates = [item for item in all_items if item not in user_item_enc]
    inputs = pd.DataFrame({'user': [user_enc] * len(candidates), 'item': candidates})

    preds = model.predict(inputs[['user', 'item']])
    top_indices = np.argsort(preds)[::-1][:top_k]
    top_items_enc = np.array(candidates)[top_indices]
    top_items = le_item.inverse_transform(top_items_enc)
    return top_items

# ----------------------------
# 6. Run Everything
# ----------------------------
print("🎯 Welcome to the Interactive Recommendation System 🎯")
print("Dataset is being loaded.")
df = load_data()
# Create a mapping from StockCode to Description
item_description_map = df.drop_duplicates(subset='StockCode').set_index('StockCode')['Description'].to_dict()

print("Data about to be processed.")
processed_df = preprocess_data(df)
processed_df, le_user, le_item = encode_features(processed_df)
print("Model is being trained.")
all_items = processed_df['item'].unique()
train_df = generate_negative_samples(processed_df, all_items, num_negatives=3)
model = train_model(train_df)

# Example usage:
while True:
    question = input("Enter CustomerID or type 'example' for demo or 'exit' to quit: ").strip()
    if question.isdigit():
        question = int(question)/1
        question = str(question)

    if question.lower() == 'exit':
        break
    elif question.lower() == 'example' or question not in le_user.classes_:
        print("Unknown user detected.\nPlease provide some details for demo recommendations.")
        print(f"Currently, {question} not implemented. Please try known user.")
        # Optional test example
        if question.lower() == 'example':
            example_user = processed_df['CustomerID'].iloc[0]
            top_recommendations = recommend_items(model, example_user, le_user, le_item, all_items, df, top_k=5)
            print("Top recommendations for user", example_user, ":", top_recommendations)
            for item in recs:
                description = item_description_map.get(item, "No description available")
                print(f" - {item}: {description}")

    else:
        recs = recommend_items(model, question, le_user, le_item, all_items, df, top_k=5)
        if len(recs) == 0:
            print("No recommendations found.")
        else:
            print(f"Top recommendations for user {question}:")
            for item in recs:
                description = item_description_map.get(item, "No description available")
                print(f" - {item}: {description}")



