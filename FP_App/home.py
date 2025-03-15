# Import needed libraries
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


# Page configuration
st.set_page_config(page_title="FakeBreaker", page_icon="📰")

# UI header
st.title("📰 FakeBreaker")
st.write("Enter a news article below to check if it's **Fake, Real, or AI-generated**.")

# Load model & tokenizer
model_path = "yasmine-11/distilbert_fake_news"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model.eval()

# Download required NLTK resources
nltk.download('punkt_tab')
# nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Text cleaning functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    """Cleans text by removing URLs, special characters, and normalizing."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in stop_words]
    return " ".join(words)  # Rejoin words

def extract_keywords(text):
    """Extracts important words (nouns & verbs) from the text."""
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    keywords = [word for word, tag in pos_tags if tag.startswith("NN") or tag.startswith("VB")]  # Keep nouns & verbs
    return ", ".join(keywords[:3])  # Return top 3 keywords

# Ensure session state is initialized
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# Text input
user_input = st.text_area(
    "📝 Article Text:", 
    value=st.session_state.user_text if "user_text" in st.session_state and st.session_state.user_text else "",
    height=250,
    placeholder="Paste the news article here..."
)

# Button layout
col1, col2 = st.columns([7, 1]) 

# Clear Text button (default style)
with col1:
    if st.button("Clear Text", key="clear_button"):
        st.session_state.user_text = None  # Reset session state value
        st.rerun()  # Properly clear input field

# Custom styling for the Analyze button
with col2:
    with stylable_container(
        "analyze_button_container",  # Unique ID for styling
        css_styles="""
        button {
            background-color: red !important;
            color: white !important;
            border-radius: 8px !important;
            border: 2px solid white !important;
            font-size: 16px !important;
            font-weight: bold !important;
            padding: 8px 20px !important;
            margin: 4px 0px !important;
        }
        button:hover {
            background-color: darkred !important;
            border-color: white !important;
        }
        """,
    ):
        analyze_button = st.button("Analyze", key="analyze_button")

# Analyze button logic
if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        st.session_state.user_text = user_input  

        with st.spinner("Analyzing... Please wait."):
            # Clean text
            cleaned_text = clean_text(user_input)

            # Tokenize & predict
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)[0]  

            # Ensure correct label mapping
            class_labels = {0: "AI-generated", 1: "Fake", 2: "Real"}
            predicted_index = torch.argmax(probs).item()

        # Display results 
        st.subheader("🧐 Model Prediction:")
        st.write(f"🔹 The article is most likely: **{class_labels[predicted_index]}**")

        st.subheader("📊 Prediction Scores:")
        for idx, score in enumerate(probs):
            st.write(f"**{class_labels[idx]}:** {score:.2%}")

        # Extract keywords
        key_terms = extract_keywords(user_input)

        # Display explanation
        st.subheader("📢 Explanation:")
        st.write(f"Key terms influencing this decision: **{key_terms}**")

        st.success("✅ Analysis complete!")
