# -*- coding: utf-8 -*-
"""
TAMU Org Matcher — Interactive Sequential Recommendation System
--------------------------------------------------------------
This script builds an adaptive ML recommender system that shows
one organization at a time and learns from user feedback.

Features:
- TF-IDF keyword similarity
- SentenceTransformer semantic similarity
- Sequential user feedback loop
- Adaptive learning with Logistic Regression
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class InteractiveOrgRecommender:
    def __init__(self, csv_file="tamu_orgs_clean.csv"):
        """Initialize the recommender system"""
        self.orgs = pd.read_csv(csv_file)
        self.orgs["tags"] = self.orgs["tags"].fillna("")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = None
        self.user_feedback = {}
        self.shown_indices = set()
        self.orgs_sorted = None
        
        # Load semantic model
        print("Loading semantic model (this may take ~10s the first time)...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!\n")
    
    def compute_initial_scores(self, user_interests):
        """Compute initial hybrid scores based on user interests"""
        # TF-IDF Similarity
        tfidf_matrix = self.vectorizer.fit_transform([user_interests] + self.orgs["tags"].tolist())
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        self.orgs["tfidf_score"] = tfidf_scores
        
        # Semantic Similarity
        user_emb = self.semantic_model.encode(user_interests, convert_to_tensor=True)
        org_embs = self.semantic_model.encode(self.orgs["description"].tolist(), convert_to_tensor=True)
        semantic_scores = util.cos_sim(user_emb, org_embs)[0]
        self.orgs["semantic_score"] = semantic_scores.cpu().numpy()
        
        # Combine scores (hybrid)
        self.orgs["final_score"] = 0.6 * self.orgs["tfidf_score"] + 0.4 * self.orgs["semantic_score"]
        self.orgs_sorted = self.orgs.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    
    def update_adaptive_scores(self):
        """Update recommendations based on user feedback using ML"""
        if len(self.user_feedback) < 2:  # Need at least 2 feedback points
            return
        
        try:
            # Get feedback data
            feedback_indices = list(self.user_feedback.keys())
            feedback_df = self.orgs_sorted.iloc[feedback_indices].copy()
            feedback_df["liked"] = [self.user_feedback[idx] for idx in feedback_indices]
            
            # Create feature matrix from tags
            all_tags = " ".join(self.orgs_sorted["tags"]).split()
            keywords = sorted(set(tag for tag in all_tags if tag.strip()))
            
            if not keywords:  # No valid tags found
                return
            
            # Build feature matrix
            X_feedback = []
            for _, row in feedback_df.iterrows():
                row_features = [1 if kw in row["tags"] else 0 for kw in keywords]
                X_feedback.append(row_features)
            
            X_feedback = np.array(X_feedback)
            y_feedback = feedback_df["liked"].to_numpy()
            
            # Train logistic regression model
            if len(set(y_feedback)) > 1:  # Need both positive and negative examples
                self.model = LogisticRegression(random_state=42)
                self.model.fit(X_feedback, y_feedback)
                
                # Predict preferences for all orgs
                X_all = []
                for _, row in self.orgs_sorted.iterrows():
                    row_features = [1 if kw in row["tags"] else 0 for kw in keywords]
                    X_all.append(row_features)
                
                X_all = np.array(X_all)
                learned_scores = self.model.predict_proba(X_all)[:, 1]
                
                # Update final scores with learned preferences
                self.orgs_sorted["learned_pref_score"] = learned_scores
                self.orgs_sorted["final_adaptive_score"] = (
                    0.5 * self.orgs_sorted["final_score"] + 
                    0.5 * self.orgs_sorted["learned_pref_score"]
                )
                
                # Re-sort based on adaptive scores
                self.orgs_sorted = self.orgs_sorted.sort_values(
                    by="final_adaptive_score", ascending=False
                ).reset_index(drop=True)
                
                print("🧠 Updated recommendations based on your feedback!")
                
        except Exception as e:
            print(f"Note: Could not update adaptive scores ({str(e)})")
    
    def get_next_recommendation(self):
        """Get the next highest-scored organization that hasn't been shown"""
        for idx in range(len(self.orgs_sorted)):
            if idx not in self.shown_indices:
                return idx, self.orgs_sorted.iloc[idx]
        return None, None
    
    def run_interactive_session(self):
        """Main interactive recommendation loop"""
        # Get user interests
        user_interests = input("Enter your interests (e.g., robotics, AI, volunteering): ").lower().strip()
        if not user_interests:
            print("No interests provided. Exiting...")
            return
        
        # Compute initial scores
        print("\n🔍 Computing recommendations...")
        self.compute_initial_scores(user_interests)
        
        print(f"\n🎯 Found {len(self.orgs_sorted)} organizations to match with your interests!")
        print("Let's find the perfect organizations for you, one at a time.\n")
        print("=" * 60)
        
        recommendation_count = 0
        
        # Interactive feedback loop
        while len(self.shown_indices) < len(self.orgs_sorted):
            # Get next recommendation
            idx, org = self.get_next_recommendation()
            if org is None:
                break
            
            recommendation_count += 1
            
            # Display recommendation
            score_type = "Adaptive Score" if hasattr(org, 'final_adaptive_score') and pd.notna(org.get('final_adaptive_score')) else "Initial Score"
            current_score = org.get('final_adaptive_score', org['final_score'])
            
            print(f"\n📌 Recommendation #{recommendation_count}")
            print(f"Organization: {org['name']}")
            print(f"Description: {org['description']}")
            print(f"{score_type}: {current_score:.3f}")
            print("-" * 50)
            
            # Get user feedback
            while True:
                feedback = input("Do you like this organization? (y/n) or 'q' to quit: ").strip().lower()
                if feedback in ['y', 'n', 'q']:
                    break
                print("Please enter 'y' for yes, 'n' for no, or 'q' to quit.")
            
            if feedback == 'q':
                print("\n👋 Thanks for using the TAMU Org Matcher!")
                break
            
            # Store feedback
            self.user_feedback[idx] = 1 if feedback == 'y' else 0
            self.shown_indices.add(idx)
            
            # Provide feedback response
            if feedback == 'y':
                print(f"✅ Great! Added '{org['name']}' to your liked organizations.")
            else:
                print(f"❌ Noted. We'll find better matches for you.")
            
            # Update model every few feedback points
            if len(self.user_feedback) >= 3 and len(self.user_feedback) % 2 == 0:
                self.update_adaptive_scores()
            
            # Ask if user wants to continue
            if recommendation_count >= 5:
                continue_session = input(f"\nYou've seen {recommendation_count} recommendations. Continue? (y/n): ").strip().lower()
                if continue_session != 'y':
                    break
        
        # Display summary
        self.display_summary()
    
    def display_summary(self):
        """Display final summary of the session"""
        liked_orgs = [self.orgs_sorted.iloc[idx]['name'] for idx, liked in self.user_feedback.items() if liked == 1]
        
        print("\n" + "=" * 60)
        print("🎉 SESSION SUMMARY")
        print("=" * 60)
        print(f"Total organizations reviewed: {len(self.user_feedback)}")
        print(f"Organizations you liked: {len(liked_orgs)}")
        
        if liked_orgs:
            print("\n✅ Your liked organizations:")
            for i, org_name in enumerate(liked_orgs, 1):
                print(f"   {i}. {org_name}")
        
        if self.model is not None:
            print(f"\n🧠 ML Model Status: Trained on {len(self.user_feedback)} feedback points")
        else:
            print(f"\n🧠 ML Model Status: Needs more feedback to train")
        
        print("\n🚀 Next steps: Reach out to your liked organizations to get involved!")
        print("=" * 60)

def main():
    """Main function to run the interactive recommender"""
    print("=" * 60)
    print("🎓 TAMU Organization Matcher - Interactive Version")
    print("=" * 60)
    print("This system will show you organizations one at a time")
    print("and learn your preferences as you provide feedback!")
    print("=" * 60)
    
    try:
        recommender = InteractiveOrgRecommender("tamu_orgs_clean.csv")
        recommender.run_interactive_session()
    except FileNotFoundError:
        print("❌ Error: Could not find '100clubs.csv' file.")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()