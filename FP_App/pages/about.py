import streamlit as st    

# Set proper page name & icon
st.set_page_config(page_title="About FakeBreaker", page_icon="ℹ️")

st.title("ℹ️ About FakeBreaker")
st.write("""
    This application uses a **DistilBERT** model to classify news articles into:
    - **Real** 
    - **Fake**
    - **AI-generated**
     
    ### 📌 How It Works:
    1. Paste the news article into the text box.
    2. Click **Analyze** to check whether it's **Real, Fake, or AI-generated**.
    3. The model, trained on the **WELFake & GPT2-output** datasets, predicts with **97% accuracy**.

    ### 🧠 About the Model:
    - **Base Model:** DistilBERT (Lightweight BERT version)
    - **Trained On:** Fake & real news datasets + AI-generated content
    - **Fine-tuned:** For multi-class classification

    ### 🚀 Future Improvements:
    - Enhancing dataset diversity
    - Improving explainability of predictions
    - Enhansing Model Generalisation
    """)
