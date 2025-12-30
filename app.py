import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import nltk
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CardioPredict Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Professional Medical UI Style ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e8f4f8 0%, #d1e7dd 50%, #f0f8f0 100%);
        background-image: 
        radial-gradient(circle at 10% 20%, rgba(33, 150, 243, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 90% 80%, rgba(76, 175, 80, 0.05) 0%, transparent 50%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main content background */
    .main .block-container {
        background-color: transparent;
        padding: 2rem 3rem;
        border-radius: 0;
        margin-top: 0;
        max-width: 100%;
    }
    
    /* Medical Banner Header */
    .medical-banner {
        background: linear-gradient(135deg, #e3f2fd 0%, #f5f5f5 100%);
        padding: 2.5rem 2rem;
        border-radius: 0;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
        border-bottom: 3px solid #1565c0;
    }
    
    .medical-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
        radial-gradient(circle at 20% 50%, rgba(33, 150, 243, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(76, 175, 80, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 20%, rgba(144, 202, 249, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .medical-banner::after {
        content: '🩺 ❤️ 💊 🏥';
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 2rem;
        opacity: 0.1;
        pointer-events: none;
    }
    
    .medical-banner-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }
    
    .medical-banner h1 {
        color: #1565c0 !important;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.8);
    }
    
    .medical-banner p {
        color: #2e7d32 !important;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .nav-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .nav-btn {
        background: white;
        color: #1565c0;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        border: 2px solid #1565c0;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .nav-btn:hover {
        background: #1565c0;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.3);
    }
    
    .nav-btn.active {
        background: #e53935;
        color: white;
        border-color: #e53935;
    }
    
    .main-header {
        background: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
        color: #2c3e50 !important;
    }
    
    .main-header h1 {
        color: #2c3e50 !important;
        margin: 0;
    }
    
    .main-header p {
        color: #666 !important;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #1565c0;
        color: #2c3e50 !important;
    }
    
    .metric-card h3 {
        color: #2c3e50 !important;
        margin-top: 0;
    }
    
    .metric-card p {
        color: #666 !important;
        margin-bottom: 0;
    }
    
    .prediction-result {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 2rem 0;
        color: #2c3e50 !important;
        border: 1px solid #e0e0e0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(21, 101, 192, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        box-shadow: 0 4px 8px rgba(21, 101, 192, 0.3);
    }
    
    /* Professional Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        padding: 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 14px 28px;
        border: none;
        border-bottom: 3px solid transparent;
        color: #666;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.2s ease;
        margin-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(33, 150, 243, 0.05);
        color: #1565c0;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #1565c0 !important;
        font-weight: 600 !important;
        border-bottom: 3px solid #e53935 !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: white;
        border-radius: 0;
        padding: 2rem;
        box-shadow: none;
        margin-top: 0;
    }
    
    /* Main content text colors */
    h1, h2, h3, h4, h5, h6 { 
        color: #2c3e50 !important; 
        font-weight: 600; 
    }
    
    .stExpander { 
        background: white; 
        border-radius: 10px; 
    }
    
    /* Fix white text visibility */
    .stMarkdown, .stText, p {
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Medical Banner Header ---
st.markdown("""
<div class="medical-banner">
    <div class="medical-banner-content">
        <h1>🏥 CardioPredict Pro</h1>
        <p>Clinical Decision Support System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load the dataset (silently)
try:
    df = pd.read_csv(r"heart_disease_uci.csv")
except:
    st.error("⚠️ System Error: Medical database not found. Please contact administrator.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("### 📊 Dataset Overview")
section = st.sidebar.radio("Select Information:",
        ["Dataset Preview","Dataset Statistics","Summary Analytics"])

if section == "Dataset Preview":
    view_option = st.sidebar.radio("Dataset View:",["Hide","Show"])
    if view_option == 'Show':
        st.sidebar.markdown("#### 📋 Sample Data")
        st.sidebar.dataframe(df.head(), use_container_width=True)

elif section == "Dataset Statistics":
    st.sidebar.markdown("#### 📈 Dataset Info")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Records", df.shape[0])
    col2.metric("Features", df.shape[1])

elif section == "Summary Analytics":
    with st.sidebar.expander("📊 Statistical Summary", expanded=False):
      st.write(df.describe())

# --- Load Resources ---
with open("corpus.pkl", "rb") as f:
    qa_corpus = pickle.load(f)

@st.cache_resource
def download_nltk_data():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("punkt_tab")
    nltk.download("omw-1.4")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")

download_nltk_data()

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# --- TF-IDF Setup ---
questions = [q for q, _ in qa_corpus]
vectorizer = TfidfVectorizer(stop_words='english', analyzer='char', ngram_range=(3, 5))
corpus_vectors = vectorizer.fit_transform(questions)

# --- NLP Preprocessing ---
def process_text(user_query):
    tokens = [w for w in word_tokenize(user_query.lower()) if w.isalnum()]
    stemmed = [stemmer.stem(w) for w in tokens]
    lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in tokens]
    pos_tags = pos_tag(tokens)
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    return stemmed, lemmatized, pos_tags, bigrams, trigrams

# --- Smart Chatbot ---
def get_response(user_query):
    query_vector = vectorizer.transform([user_query])
    sims = cosine_similarity(query_vector, corpus_vectors).flatten()
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score > 0.1:
        answer = qa_corpus[best_idx][1]
    else:
        answer = "Sorry, I don't have an answer for that. Please ask another question!"

    return answer, best_score

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Analysis", "💬 Chatbot", "🎯 Predict"])

# ========== TAB 1: HOME ==========
with tab1:
    st.markdown("""
    <div class="metric-card">
        <h3>Welcome to CardioPredict Pro</h3>
        <p><b>CardioPredict Pro</b> is a machine learning-powered clinical decision support tool that assesses cardiovascular risk using validated medical parameters. 
        The system employs an ensemble of 8 ML algorithms to provide evidence-based risk stratification for healthcare professionals.</p>
        <h4>Key Features:</h4>
        <ul>
        <li><b>📊 Data Analytics:</b> Comprehensive analysis of patient data with interactive visualizations</li>
        <li><b>💬 AI Chatbot:</b> Get instant answers to health-related questions</li>
        <li><b>🎯 Risk Prediction:</b> Multi-model ensemble prediction for cardiovascular disease risk assessment</li>
        <li><b>📈 Clinical Insights:</b> Detailed statistical analysis and patient demographics</li>
        </ul>
        <p><b>How to Use:</b> Navigate through the tabs above to access different features of the application.</p>
    </div>
    """, unsafe_allow_html=True)

# ========== TAB 2: ANALYSIS ==========
with tab2:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Clinical Data Analytics</h3>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- Q1: Age Distribution ----------------
    with st.expander("**Q1: What is the age distribution of patients?**", expanded=True):
        st.markdown(
        "**Description**: Histogram showing the age distribution of patients. "
        "Most patients fall in the 40–60 age range, indicating higher risk in middle age."
        )

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.hist(df["age"], bins=20, color="lightgreen", edgecolor="black")
        ax.set_title("Age Distribution of Patients")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Patients")
        ax.grid(False)

        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        st.pyplot(fig)

    # ---------------- Q2: Gender Distribution ----------------
    with st.expander("**Q2: What is the gender distribution of patients?**"):
        gender_count = df["sex"].value_counts()

        st.markdown(
        f"**Description**: Donut chart showing gender distribution. "
        f"The majority of patients are **{gender_count.idxmax()}**."
        )

        # Very small figure
        fig, ax = plt.subplots(figsize=(3, 3))

        wedges, texts, autotexts = ax.pie(
        gender_count.values,
        labels=gender_count.index,
        autopct='%1.0f%%',
        startangle=90,
        textprops={'fontsize': 8},
        wedgeprops={'width': 0.35}  # 🔥 Donut thickness
        )

        # Center circle (donut hole)
        centre_circle = plt.Circle((0, 0), 0.55, fc='none')
        ax.add_artist(centre_circle)

        ax.set_title("Gender Distribution", fontsize=10)

        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        ax.axis('equal')

        # 🚨 Critical for small size in Streamlit
        st.pyplot(fig, use_container_width=False)

    # ---------------- Q3: Chest Pain Type Distribution ----------------
    with st.expander("**Q3: What type of chest pain is most common among patients?**"):
        cp_count = df["cp"].value_counts()
        top_cp = cp_count.idxmax()

        st.markdown(
        f"**Description**: Distribution of chest pain types. "
        f"The most common type is **{top_cp}**, often linked to heart-related issues."
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
        cp_count.values,
        labels=cp_count.index,
        autopct='%1.1f%%',
        startangle=90
        )
        ax.set_title("Chest Pain Type Distribution")

        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        st.pyplot(fig)

    # ---------------- Q4: Cholesterol Levels ----------------
    with st.expander("**Q4: What is the distribution of cholesterol levels?**"):
        st.markdown(
        "**Description**: Histogram representing cholesterol levels. "
        "Many patients show cholesterol values between 200–300 mg/dl."
        )

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.hist(df["chol"], bins=20, color="lightblue", edgecolor="black")
        ax.set_title("Cholesterol Level Distribution")
        ax.set_xlabel("Cholesterol (mg/dl)")
        ax.set_ylabel("Number of Patients")
        ax.grid(False)

        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        st.pyplot(fig)

    # ---------------- Q5: Heart Disease Presence ----------------
    with st.expander("**Q5: How many patients have heart disease?**"):
        status = df["num"].gt(0).map({True: "Disease", False: "No Disease"})
        counts = status.value_counts()

        st.markdown(
        f"**Description**: Pie chart showing heart disease presence. "
        f"Most patients fall under **{counts.idxmax()}**."
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Heart Disease Presence")

        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        st.pyplot(fig)

# ========== TAB 3: CHATBOT ==========
with tab3:
    st.markdown("""
    <div class="metric-card">
        <h3>💬 AI Health Assistant</h3>
        <p>Get instant answers to your health-related questions</p>
    </div>
    """, unsafe_allow_html=True)

    # Dropdown for predefined questions
    dropdown_question = st.selectbox(
        "Select a predefined question (optional):",
        ["-- Select a question --"] + questions
    )

    # Text input for custom question
    user_question = st.text_input("Or type your own question:")

    # Determine which question to use (user input takes priority)
    final_question = None
    if user_question.strip():
        final_question = user_question.strip()
    elif dropdown_question != "-- Select a question --":
        final_question = dropdown_question

    if st.button("Get Answer"):
        if final_question:
           answer, score = get_response(final_question)
           stemmed, lemmatized, pos_tags, bigrams, trigrams = process_text(final_question)
           st.success(f"Answer: {answer}")
           st.caption(f"🎯 Similarity Score: {score:.2f}")
        else:
           st.warning("Please select or type a question!")

# ========== TAB 4: PREDICT ==========
with tab4:
    # --- Load Models ---
    knn = joblib.load("knn_model.pkl")
    log = joblib.load("logistic_model.pkl")
    gb = joblib.load("Gaussian_naivebayes_model.pkl")
    dt = joblib.load("DecisionTree_model.pkl")
    svc = joblib.load("svc_model.pkl")
    xg = joblib.load("XGBoost_model.pkl")
    cbc = joblib.load("CatBoost_model.pkl")
    lgbm = joblib.load("lightboost_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # --- Patient Assessment Form ---
    st.markdown("""
    <div class="metric-card">
        <h3>🩺 Patient Assessment</h3>
        <p><b>Instructions:</b> Please enter the patient's medical parameters below. Use actual clinical values from recent medical examinations. 
        Normal ranges are provided as guidance - consult medical records for accurate values.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Enter Clinical Parameters")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🫀 Vital Signs & Demographics**")
        age = st.slider("👤 Age (years)", 20, 90, 50)
        st.markdown("**📝 Patient's current age in years**")
        trestbps = st.slider("🩸 Resting Blood Pressure (mmHg)", 80, 200, 120)
        st.markdown("**📝 Systolic blood pressure at rest. Normal: 90-140 mmHg**")
        chol = st.slider("🧪 Serum Cholesterol (mg/dl)", 100, 400, 200)
        st.markdown("**📝 Total cholesterol level. Normal: <200 mg/dl, High: >240 mg/dl**")
        thalch = st.slider("💓 Maximum Heart Rate Achieved", 70, 210, 150)
        st.markdown("**📝 Highest heart rate during exercise stress test**")
        oldpeak = st.slider("📈 ST Depression (Oldpeak)", 0.0, 6.5, 1.0)
        st.markdown("**📝 ST depression induced by exercise. Higher values indicate more risk**")
        fbs = st.radio("🍯 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        st.markdown("**📝 Blood sugar after 8+ hours fasting. Yes if >120 mg/dl**")
        fbs = 1 if fbs == "Yes" else 0

    with col2:
        st.markdown("**🏥 Clinical Assessments**")
        sex = st.selectbox("⚧ Gender", ["Male", "Female"])
        st.markdown("**📝 Biological gender affects heart disease risk patterns**")
        sex = 1 if sex == "Male" else 0

        cp = st.selectbox("💔 Chest Pain Type", 
                        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        st.markdown("**📝 Type of chest pain experienced: Typical Angina = classic heart-related pain**")
        cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

        restecg = st.selectbox("📊 Resting ECG Results", 
                            ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        st.markdown("**📝 Electrocardiogram results at rest. LV Hypertrophy = enlarged left ventricle**")
        restecg = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)

        exang = st.radio("🏃 Exercise Induced Angina", ["No", "Yes"])
        st.markdown("**📝 Does exercise trigger chest pain? Yes indicates higher risk**")
        exang = 1 if exang == "Yes" else 0

        slope = st.selectbox("📉 ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
        st.markdown("**📝 Slope of ST segment during peak exercise. Downsloping = higher risk**")
        slope = ["Upsloping", "Flat", "Downsloping"].index(slope)

        ca = st.selectbox("🫀 Major Vessels (0-3)", [0, 1, 2, 3])
        st.markdown("**📝 Number of major coronary vessels with >50% narrowing (from angiography)**")

        thal = st.selectbox("🩺 Thalassemia Status", ["Normal", "Fixed Defect", "Reversible Defect"])
        st.markdown("**📝 Blood disorder status. Fixed/Reversible defects indicate higher risk**")
        thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

    # --- Prepare Input ---
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalch, exang, oldpeak, slope, ca, thal]])

    features = scaler.transform(features)

    # --- Risk Assessment Results ---
    if st.button("🎯 Analyze Risk", use_container_width=True):
        models = {
            "KNN": knn, "Logistic Regression": log, "Gaussian NB": gb,
            "Decision Tree": dt, "SVC": svc, "XGBoost": xg,
            "CatBoost": cbc, "LightGBM": lgbm
        }

        preds = {name: int(model.predict(features)[0]) for name, model in models.items()}
        final_pred = max(set(preds.values()), key=list(preds.values()).count)

        # Professional results display
        st.markdown("""
        <div class="prediction-result">
            <h2>🎯 Risk Assessment Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col2:
            if final_pred == 0:
                st.success("💚 Low Risk - No Heart Disease Detected")
            else:
                st.error("❤️ High Risk - Heart Disease Indicators Present")
        
        # Model predictions in columns
        st.markdown("#### 🧠 Model Consensus")
        cols = st.columns(4)
        for i, (name, pred) in enumerate(preds.items()):
            with cols[i % 4]:
                st.metric(name, "Risk" if pred > 0 else "Normal")

        st.download_button(
            label="📥 Download Report",
            data=f"Patient Risk Assessment\nResult: {'High Risk' if final_pred > 0 else 'Low Risk'}\nModels: {preds}",
            file_name="heart_risk_report.txt",
            mime="text/plain",
            use_container_width=True
        )


