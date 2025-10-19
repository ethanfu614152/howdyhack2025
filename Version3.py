import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
import numpy as np

def load_org_data(csv_file):
    orgs = pd.read_csv(csv_file)
    orgs["tags"] = orgs["tags"].fillna("")
    return orgs

def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_initial_scores(orgs, user_interests, vectorizer, semantic_model):
    tfidf_matrix = vectorizer.fit_transform([user_interests] + orgs["tags"].tolist())
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    orgs = orgs.copy()
    orgs["tfidf_score"] = tfidf_scores

    user_emb = semantic_model.encode(user_interests, convert_to_tensor=True)
    org_embs = semantic_model.encode(orgs["description"].tolist(), convert_to_tensor=True)
    semantic_scores = util.cos_sim(user_emb, org_embs)[0]
    orgs["semantic_score"] = semantic_scores.cpu().numpy()

    orgs["final_score"] = 0.6 * orgs["tfidf_score"] + 0.4 * orgs["semantic_score"]
    orgs_sorted = orgs.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    return orgs_sorted

def update_adaptive_scores(orgs_sorted, user_feedback):
    if len(user_feedback) < 2:
        return orgs_sorted, None

    feedback_indices = list(user_feedback.keys())
    feedback_df = orgs_sorted.iloc[feedback_indices].copy()
    feedback_df["liked"] = [user_feedback[idx] for idx in feedback_indices]

    all_tags = " ".join(orgs_sorted["tags"]).split()
    keywords = sorted(set(tag for tag in all_tags if tag.strip()))
    if not keywords:
        return orgs_sorted, None

    X_feedback = []
    for _, row in feedback_df.iterrows():
        row_features = [1 if kw in row["tags"] else 0 for kw in keywords]
        X_feedback.append(row_features)
    X_feedback = np.array(X_feedback)
    y_feedback = feedback_df["liked"].to_numpy()

    if len(set(y_feedback)) > 1:
        model = LogisticRegression(random_state=42)
        model.fit(X_feedback, y_feedback)

        X_all = []
        for _, row in orgs_sorted.iterrows():
            row_features = [1 if kw in row["tags"] else 0 for kw in keywords]
            X_all.append(row_features)
        X_all = np.array(X_all)
        learned_scores = model.predict_proba(X_all)[:, 1]

        orgs_sorted = orgs_sorted.copy()
        orgs_sorted["learned_pref_score"] = learned_scores
        orgs_sorted["final_adaptive_score"] = (
            0.5 * orgs_sorted["final_score"] +
            0.5 * orgs_sorted["learned_pref_score"]
        )
        orgs_sorted = orgs_sorted.sort_values(
            by="final_adaptive_score", ascending=False
        ).reset_index(drop=True)
        return orgs_sorted, model
    else:
        return orgs_sorted, None

def get_next_recommendation(orgs_sorted, shown_indices):
    for idx in range(len(orgs_sorted)):
        if idx not in shown_indices:
            return idx, orgs_sorted.iloc[idx]
    return None, None
#new
def record_feedback(user_feedback, shown_indices, idx, liked):
    user_feedback[idx] = 1 if liked else 0
    shown_indices.add(idx)
    return user_feedback, shown_indices

def get_liked_orgs(orgs_sorted, user_feedback):
    return [orgs_sorted.iloc[idx]['name'] for idx, liked in user_feedback.items() if liked == 1]

def get_summary(orgs_sorted, user_feedback, model):
    liked_orgs = get_liked_orgs(orgs_sorted, user_feedback)
    summary = {
        "total_reviewed": len(user_feedback),
        "liked_count": len(liked_orgs),
        "liked_orgs": liked_orgs,
        "ml_status": f"Trained on {len(user_feedback)} feedback points" if model is not None else "Needs more feedback to train"
    }
    return summary

# Assume all modular functions are imported from your_module

# Step 1: Load data and models
orgs = load_org_data("tamu_orgs_clean.csv")
vectorizer = TfidfVectorizer(stop_words='english')
semantic_model = load_semantic_model()

# Step 2: Get user interests
user_interests = st.text_input("Enter your interests:")

# Step 3: Compute initial scores
orgs_sorted = compute_initial_scores(orgs, user_interests, vectorizer, semantic_model)

# Step 4: Initialize feedback and shown indices
user_feedback = {}
shown_indices = set()
model = None

# Step 5: Show recommendations and record feedback
for _ in range(2):  # Show two recommendations for demo
    idx, org = get_next_recommendation(orgs_sorted, shown_indices)
    if org is None:
        break
    print(f"Recommended: {org['name']} - {org['description']}")
    # Simulate user feedback (like first, dislike second)
    liked = True if _ == 0 else False
    user_feedback, shown_indices = record_feedback(user_feedback, shown_indices, idx, liked)

# Step 6: Update adaptive scores after feedback
orgs_sorted, model = update_adaptive_scores(orgs_sorted, user_feedback)

# Step 7: Get liked organizations
liked_orgs = get_liked_orgs(orgs_sorted, user_feedback)
print("Liked organizations:", liked_orgs)

# Step 8: Get summary
summary = get_summary(orgs_sorted, user_feedback, model)
print("Session summary:", summary)
