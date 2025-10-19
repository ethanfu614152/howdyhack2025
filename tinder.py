import streamlit as st
import pandas as pd

st.header("AggieBanner")
st.caption("This will do something idk")


# Define your card data
CARD_DATA = [
    {"info": "Epic Movement: We provide a safe space where students can join and interact in a community of believers who are set on living out the Gospel of Jesus Christ. Our goal is to create a learning and social setting for students through weekly Bible studies, large group activities, and leadership opportunities. While our focus is on Asian-American students, anyone is encouraged to participate in our activities and/or join, especially those that want to learn more about Christ.", "key": "c1"},
    {"info": "AAVI TAMU: we suck", "key": "c2"},
    {"info": "Card 3: Final check, are you ready?", "key": "c3"},
    # Add more cards...
]

# Initialize state
if 'current_card_index' not in st.session_state:
    st.session_state.current_card_index = 0
if 'results' not in st.session_state:
    st.session_state.results = {}



def advance_card(choice):
    # Record the choice
    current_key = CARD_DATA[st.session_state.current_card_index]['key']
    st.session_state.results[current_key] = choice
    
    # Move to the next card
    st.session_state.current_card_index += 1



card_index = st.session_state.current_card_index

if card_index < len(CARD_DATA):
    # Get the data for the current card
    current_card = CARD_DATA[card_index]
    
    # --- Card Display (Styling with st.container and st.markdown for a 'card' look) ---
    with st.container(border=True):
        st.subheader(f"Card {card_index + 1}")
        st.write(current_card['info'])
        
        # --- Buttons ---
        col1, col2 = st.columns(2)
        
        with col1:
            # The 'on_click' uses a lambda or a function to pass arguments
            st.button("✅ Accept", 
                      type="primary",
                      use_container_width=True,
                      on_click=advance_card,
                      args=("accept",)) # Pass 'accept' to the callback
        
        with col2:
            st.button("❌ Reject", 
                      type="secondary",
                      use_container_width=True,
                      on_click=advance_card,
                      args=("reject",)) # Pass 'reject' to the callback
            
    # --- Animation/Visual effect (Simple loading spinner or success/error for transition) ---
    # Streamlit doesn't have native "slide" animations, but a short visual can imply transition.
    if card_index > 0 and 'last_card_index' in st.session_state and st.session_state.last_card_index < card_index:
        # This will show a quick 'transition' message after a choice is made and before the new card loads
        st.toast(f"Moving to Card {card_index + 1}...")

    # Update the last card index for the next run
    st.session_state.last_card_index = card_index
    
else:
    # --- Results Display ---
    st.success("🎉 All cards reviewed! Here are your results:")
    st.json(st.session_state.results)