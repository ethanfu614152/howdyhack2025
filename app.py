import streamlit as st
import pandas as pd
import numpy as np
from your_recommender_module import InteractiveOrgRecommender  # Your class from the script provided

def initialize_recommender():
    recommender = InteractiveOrgRecommender("tamu_orgs_clean.csv")
    recommender.user_feedback = {}
    recommender.shown_indices = set()
    recommender.orgs_sorted = None
    return recommender

def main():
    st.title("🎓 TAMU Organization Matcher")

    # Initialize recommender once using session_state
    if "recommender" not in st.session_state:
        st.session_state.recommender = initialize_recommender()
        st.session_state.current_idx = 0
        st.session_state.scores_computed = False
        st.session_state.updated = False

    recommender = st.session_state.recommender

    # Input for user interests
    interests = st.text_input("Enter your interests (e.g., robotics, AI, volunteering):")

    # Compute initial scores when interests provided
    if interests and not st.session_state.scores_computed:
        recommender.compute_initial_scores(interests.lower())
        st.session_state.orgs_sorted = recommender.orgs_sorted
        st.session_state.scores_computed = True
        st.session_state.current_idx = 0
        st.session_state.user_feedback = {}

    if st.session_state.scores_computed:
        org_list = st.session_state.orgs_sorted

        # Check if more orgs to show
        if st.session_state.current_idx < len(org_list):
            org = org_list.iloc[st.session_state.current_idx]
            
            st.subheader(f"Recommendation #{st.session_state.current_idx + 1}")
            st.markdown(f"**Organization:** {org['name']}")
            st.markdown(f"**Description:** {org['description']}")
            score = org.get('final_adaptive_score', org['final_score'])
            st.markdown(f"**Score:** {score:.3f}")

            # Buttons for feedback in columns
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Like", key=f"like{st.session_state.current_idx}"):
                    st.session_state.user_feedback[st.session_state.current_idx] = 1
                    st.session_state.current_idx += 1
                    st.experimental_rerun()

            with col2:
                if st.button("👎 Dislike", key=f"dislike{st.session_state.current_idx}"):
                    st.session_state.user_feedback[st.session_state.current_idx] = 0
                    st.session_state.current_idx += 1
                    st.experimental_rerun()

            # Update model every 2 feedback points
            if len(st.session_state.user_feedback) >= 2 and len(st.session_state.user_feedback) % 2 == 0:
                if not st.session_state.updated:
                    recommender.user_feedback = st.session_state.user_feedback
                    recommender.orgs_sorted = st.session_state.orgs_sorted
                    recommender.update_adaptive_scores()
                    st.session_state.orgs_sorted = recommender.orgs_sorted
                    st.session_state.updated = True
                    st.experimental_rerun()
            else:
                st.session_state.updated = False

        else:
            st.write("🎉 You've reviewed all recommendations!")
            liked_orgs = [st.session_state.orgs_sorted.iloc[i]['name'] for i, liked in st.session_state.user_feedback.items() if liked]
            st.write(f"You liked {len(liked_orgs)} organizations:")
            for org_name in liked_orgs:
                st.write(f"- {org_name}")

if __name__ == "__main__":
    main()