import streamlit as st
import numpy as np
import joblib, pickle, nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE STYLE --------------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://i.ibb.co/QjQHvRZf/image.jpg");
    background-size: cover;
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: absolute; inset: 0;
    background-color: rgba(0, 0, 0, 0.55);
    z-index: 0;
}
.stApp > div { position: relative; z-index: 1; color: white; }
h1, h2, h3, p, label { color: white !important; }
div.stButton > button {
    background-color: #00BFFF; color: white; border: none;
    border-radius: 10px; height: 3em; width: 25%; font-weight: bold;
    transition: 0.3s;
}
div.stButton > button:hover { background-color: #1E90FF; }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD RESOURCES --------------------
with open("corpus.pkl", "rb") as f:
    qa_corpus = pickle.load(f)

lr = joblib.load("linear_model.pkl")
ridge = joblib.load("ridge_model.pkl")
lasso = joblib.load("lasso_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------- NLP SETUP --------------------
nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')
stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()

questions = [q for q, _ in qa_corpus]
vectorizer = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(3, 5))
corpus_vectors = vectorizer.fit_transform(questions)

def process_text(text):
    tokens = [w for w in word_tokenize(text.lower()) if w.isalnum()]
    return {
        "stemmed": [stemmer.stem(w) for w in tokens],
        "lemmatized": [lemmatizer.lemmatize(w, pos='v') for w in tokens],
        "pos": pos_tag(tokens),
        "bigrams": list(ngrams(tokens, 2)),
        "trigrams": list(ngrams(tokens, 3))
    }

def get_response(query):
    sims = cosine_similarity(vectorizer.transform([query]), corpus_vectors).flatten()
    idx, score = np.argmax(sims), np.max(sims)
    ans = qa_corpus[idx][1] if score > 0.1 else "Sorry, I don't have an answer for that!"
    return ans, score

# -------------------- STREAMLIT UI --------------------
st.title("🏠 California House Application by Sanjana")

# --- House Price Prediction ---
st.subheader("📊 House Price Prediction")
inputs = {
    "MedInc": st.number_input("Median Income (10k$)", 0.0),
    "HouseAge": st.number_input("House Age", 1.0),
    "AveRooms": st.number_input("Average Rooms", 1.0),
    "AvgBedrms": st.number_input("Average Bedrooms", 0.0),
    "Population": st.number_input("Population", 1.0),
    "AveOccup": st.number_input("Average Occupancy", 1.0),
    "Latitude": st.number_input("Latitude", 32.0, 42.0),
    "Longitude": st.number_input("Longitude", -124.0, -114.0)
}

features = scaler.transform([list(inputs.values())])

if st.button("Predict House Price"):
    st.success(f"Linear Regression: ${lr.predict(features)[0]*100000:.2f}")
    st.success(f"Ridge Regression:  ${ridge.predict(features)[0]*100000:.2f}")
    st.success(f"Lasso Regression:  ${lasso.predict(features)[0]*100000:.2f}")

# --- Chatbot Section ---
st.subheader("💬 Ask about the Housing Project")

dropdown = st.selectbox("Select a predefined question (optional):", ["-- Select --"] + questions)
user_q = st.text_input("Or type your own question:")
final_q = user_q.strip() if user_q.strip() else (dropdown if dropdown != "-- Select --" else None)

if st.button("Get Answer"):
    if final_q:
        ans, score = get_response(final_q)
        nlp = process_text(final_q)
        st.success(f"Answer: {ans}")
        st.caption(f"🔍 Similarity Score: {score:.2f}")
        with st.expander("NLP Details"):
            st.write(f"**Stemmed:** {nlp['stemmed']}")
            st.write(f"**Lemmatized:** {nlp['lemmatized']}")
            st.write(f"**POS Tags:** {nlp['pos']}")
            st.write(f"**Bigrams:** {nlp['bigrams']}")
            st.write(f"**Trigrams:** {nlp['trigrams']}")
    else:
        st.warning("Please select or type a question!")
