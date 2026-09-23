import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ShieldSMS • Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# NLTK SETUP
# =========================================================

@st.cache_resource
def prepare_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    return set(nltk.corpus.stopwords.words("english"))


stop_words = prepare_nltk()


# =========================================================
# LOAD MODEL + VECTORIZER
# =========================================================

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


model, vectorizer = load_model()
ps = PorterStemmer()


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "message": "",
    "prediction": None,
    "ham_probability": None,
    "spam_probability": None,
    "transformed_message": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>
/* ---------- Global ---------- */

.stApp {
    min-height: 100vh;
    background:
        radial-gradient(circle at 8% 12%, rgba(99,102,241,.24) 0, rgba(99,102,241,.08) 18%, transparent 36%),
        radial-gradient(circle at 92% 10%, rgba(14,165,233,.20) 0, rgba(14,165,233,.07) 20%, transparent 38%),
        radial-gradient(circle at 82% 78%, rgba(168,85,247,.17) 0, rgba(168,85,247,.06) 22%, transparent 40%),
        radial-gradient(circle at 12% 88%, rgba(20,184,166,.14) 0, rgba(20,184,166,.05) 20%, transparent 38%),
        linear-gradient(135deg, #f8f7ff 0%, #eef5ff 45%, #f5f0ff 100%);
    background-attachment: fixed;
    color: #172033;
    position: relative;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(99,102,241,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,.035) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 75%);
    -webkit-mask-image: linear-gradient(to bottom, black, transparent 75%);
}

.stApp::after {
    content: "";
    position: fixed;
    width: 420px;
    height: 420px;
    right: -150px;
    bottom: -170px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,.16), transparent 68%);
    filter: blur(10px);
    pointer-events: none;
}

.block-container {
    max-width: 1180px;
    padding: 1.4rem 2rem 3rem 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

footer {
    visibility: hidden;
}


/* ---------- Top navigation ---------- */

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 2px 22px 2px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 11px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    font-size: 21px;
    box-shadow: 0 8px 20px rgba(79,70,229,.22);
}

.brand-name {
    font-weight: 800;
    font-size: 17px;
    color: #172033;
    line-height: 1.1;
}

.brand-caption {
    color: #7b8497;
    font-size: 11px;
    margin-top: 3px;
}

/* ---------- Hero ---------- */

.hero {
    text-align: center;
    padding: 42px 10px 34px;
}

.hero-title {
    margin: 0;
    font-size: clamp(38px, 5vw, 62px);
    line-height: 1.05;
    letter-spacing: -2.5px;
    font-weight: 900;
    background: linear-gradient(90deg, #4338ca, #7c3aed 48%, #0891b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 670px;
    margin: 16px auto 0;
    color: #667085;
    font-size: 16px;
    line-height: 1.7;
}


/* ---------- Main cards ---------- */

.card {
    background: rgba(255,255,255,.94);
    border: 1px solid #e5e9f2;
    border-radius: 22px;
    box-shadow: 0 18px 45px rgba(31,41,55,.07);
}

.input-card {
    padding: 25px;
    border-top: 4px solid #6366f1;
    background: rgba(255,255,255,.97);
}

.card-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 17px;
}

.card-heading-icon {
    width: 38px;
    height: 38px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eef2ff;
    color: #4f46e5;
    font-size: 18px;
}

.card-heading-title {
    font-weight: 800;
    color: #1d2939;
    font-size: 17px;
}

.card-heading-subtitle {
    color: #8a93a5;
    font-size: 12px;
    margin-top: 2px;
}


/* ---------- Text area ---------- */

textarea {
    border-radius: 16px !important;
    border: 1px solid #dfe4ec !important;
    background: #fbfcfe !important;
    color: #172033 !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
}


/* ---------- Example chips ---------- */

.example-label {
    color: #667085;
    font-size: 12px;
    font-weight: 700;
    margin: 18px 0 9px;
    text-transform: uppercase;
    letter-spacing: .06em;
}


/* ---------- Buttons ---------- */

.stButton > button {
    border-radius: 14px !important;
    min-height: 48px !important;
    font-weight: 750 !important;
    border: 1px solid #dfe4ec !important;
    background: rgba(255,255,255,.96) !important;
    color: #344054 !important;
    transition: all .18s ease !important;
    box-shadow: 0 3px 12px rgba(15,23,42,.035) !important;
}

.stButton > button:hover {
    border-color: #818cf8 !important;
    color: #4338ca !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(79,70,229,.12) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 10px 24px rgba(79,70,229,.22) !important;
}

.stButton > button[kind="primary"]:hover {
    color: white !important;
    filter: brightness(1.04);
}

.analyze-wrap {
    margin-top: 16px;
}


/* ---------- Result ---------- */

.result {
    border-radius: 22px;
    padding: 26px;
    margin-top: 22px;
    position: relative;
    overflow: hidden;
}

.result-spam {
    background: linear-gradient(135deg, #fff1f2 0%, #fff7ed 100%);
    border: 1px solid #fda4af;
    box-shadow: 0 14px 35px rgba(244,63,94,.10);
}

.result-ham {
    background: linear-gradient(135deg, #ecfdf5 0%, #eff6ff 100%);
    border: 1px solid #86efac;
    box-shadow: 0 14px 35px rgba(16,185,129,.10);
}

.result-top {
    display: flex;
    align-items: center;
    gap: 14px;
}

.result-icon {
    width: 50px;
    height: 50px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

.result-spam .result-icon {
    background: #ffe4e6;
}

.result-ham .result-icon {
    background: #d1fae5;
}

.result-title {
    font-size: 22px;
    font-weight: 850;
    color: #172033;
}

.result-subtitle {
    color: #667085;
    font-size: 13px;
    margin-top: 3px;
}


/* ---------- Colorful section labels ---------- */

.section-title {
    font-size: 19px;
    font-weight: 850;
    color: #172033;
    margin-top: 24px;
    margin-bottom: 10px;
}

.quick-label {
    color: #4f46e5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin: 18px 0 9px;
}

/* ---------- Streamlit progress ---------- */

div[data-testid="stProgress"] > div > div {
    border-radius: 99px !important;
}

/* ---------- Confidence ---------- */

.confidence-card {
    margin-top: 16px;
    padding: 20px;
    border: 1px solid #e5e9f2;
    background: white;
    border-radius: 18px;
}

.confidence-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.confidence-label {
    color: #667085;
    font-size: 13px;
    font-weight: 700;
}

.confidence-value {
    color: #172033;
    font-size: 20px;
    font-weight: 850;
}


/* ---------- Probability bars ---------- */

.prob-card {
    margin-top: 16px;
    padding: 20px;
    border: 1px solid #e5e9f2;
    background: white;
    border-radius: 18px;
}

.prob-row {
    margin-bottom: 15px;
}

.prob-header {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 700;
    color: #475467;
    margin-bottom: 7px;
}

.bar {
    width: 100%;
    height: 9px;
    background: #eef1f5;
    border-radius: 99px;
    overflow: hidden;
}

.bar-fill-ham {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    border-radius: 99px;
}

.bar-fill-spam {
    height: 100%;
    background: linear-gradient(90deg, #f43f5e, #fb7185);
    border-radius: 99px;
}


/* ---------- Stat cards ---------- */

.stat {
    background: white;
    border: 1px solid #e5e9f2;
    border-radius: 16px;
    padding: 16px 12px;
    text-align: center;
}

.stat-number {
    font-size: 22px;
    font-weight: 850;
    color: #172033;
}

.stat-label {
    font-size: 11px;
    color: #8a93a5;
    margin-top: 3px;
}


/* ---------- How it works ---------- */

.how-card {
    margin-top: 22px;
    padding: 22px;
}

.step {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin: 12px 0;
}

.step-number {
    min-width: 28px;
    height: 28px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eef2ff;
    color: #4f46e5;
    font-weight: 800;
    font-size: 12px;
}

.step-text {
    color: #667085;
    font-size: 13px;
    line-height: 1.5;
}


/* ---------- Footer ---------- */

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 12px;
    padding: 30px 0 8px;
}


/* ---------- Mobile ---------- */

@media (max-width: 900px) {
    .status-pill {
        display: none;
    }
}

@media (max-width: 700px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-title {
        letter-spacing: -1px;
    }

    .input-card {
        padding: 18px;
    }

    .status-pill {
        display: none;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# TOP NAVIGATION
# =========================================================

st.markdown(
    """
<div class="navbar">
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">ShieldSMS</div>
            <div class="brand-caption">AI message protection</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">Is this message safe?</div>
    <div class="hero-subtitle">
        Paste an SMS below and we'll check it for potential spam.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# MAIN INPUT CARD
# =========================================================

st.markdown(
    """
<div class="card input-card">
    <div class="card-heading">
        <div class="card-heading-icon">✉️</div>
        <div>
            <div class="card-heading-title">Analyze a message</div>
            <div class="card-heading-subtitle">
                Paste the complete SMS for the best result
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# Example buttons are placed immediately below the heading.
st.markdown(
    '<div class="quick-label">⚡ Quick examples</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚨 Suspicious", use_container_width=True):
        st.session_state.message = (
            "Congratulations! You have won a $1000 cash prize. "
            "Click now to claim your reward!"
        )
        st.session_state.prediction = None
        st.rerun()

with col2:
    if st.button("💬 Normal", use_container_width=True):
        st.session_state.message = (
            "Hey, are we meeting for lunch today?"
        )
        st.session_state.prediction = None
        st.rerun()

with col3:
    if st.button("🏷️ Promotion", use_container_width=True):
        st.session_state.message = (
            "Get 50% discount on your next purchase. "
            "Offer valid today only."
        )
        st.session_state.prediction = None
        st.rerun()


# =========================================================
# TEXT INPUT
# =========================================================

message = st.text_area(
    "Message",
    value=st.session_state.message,
    placeholder="Example: Hey, are we still meeting at 6 PM?",
    height=155,
    label_visibility="collapsed",
)


# Live message information.
char_count = len(message)
word_count = len(message.split()) if message.strip() else 0

st.caption(
    f"{char_count} characters  •  {word_count} words"
)


# =========================================================
# ACTIONS
# =========================================================

col1, col2 = st.columns([2, 1])

with col1:
    analyze = st.button(
        "🔍  Analyze message",
        use_container_width=True,
        type="primary",
    )

with col2:
    clear = st.button(
        "↻  Clear",
        use_container_width=True,
    )


if clear:
    st.session_state.message = ""
    st.session_state.prediction = None
    st.session_state.ham_probability = None
    st.session_state.spam_probability = None
    st.session_state.transformed_message = ""
    st.rerun()


# =========================================================
# ANALYZE
# =========================================================

if analyze:

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        with st.spinner("Analyzing message..."):

            transformed_message = transform_text(message)

            vector_input = vectorizer.transform(
                [transformed_message]
            ).toarray()

            prediction = int(model.predict(vector_input)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vector_input)[0]
                ham_probability = float(probabilities[0])
                spam_probability = float(probabilities[1])
            else:
                ham_probability = 1.0 if prediction == 0 else 0.0
                spam_probability = 1.0 if prediction == 1 else 0.0

            st.session_state.prediction = prediction
            st.session_state.ham_probability = ham_probability
            st.session_state.spam_probability = spam_probability
            st.session_state.transformed_message = transformed_message


# =========================================================
# RESULT
# =========================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction
    ham_probability = st.session_state.ham_probability
    spam_probability = st.session_state.spam_probability

    if prediction == 1:
        result_class = "result result-spam"
        icon = "🚨"
        title = "Potential spam detected"
        subtitle = "This message has patterns associated with spam."
        confidence = spam_probability
    else:
        result_class = "result result-ham"
        icon = "✓"
        title = "Looks like a legitimate message"
        subtitle = "The model classified this message as ham."
        confidence = ham_probability

    st.markdown(
        f"""
<div class="{result_class}">
    <div class="result-top">
        <div class="result-icon">{icon}</div>
        <div>
            <div class="result-title">{title}</div>
            <div class="result-subtitle">{subtitle}</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    st.markdown(
        """
<div class="confidence-card">
    <div class="confidence-row">
        <div class="confidence-label">MODEL CONFIDENCE</div>
        <div class="confidence-value">
"""
        + f"{confidence * 100:.1f}%"
        + """
        </div>
    </div>
"""
        + f'<div class="bar"><div class="bar-fill-{"spam" if prediction == 1 else "ham"}" style="width:{confidence * 100:.2f}%"></div></div>'
        + """
</div>
""",
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # Probability breakdown
    # Use native Streamlit components here instead of a large
    # HTML block, which avoids raw HTML being displayed.
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Prediction breakdown</div>',
        unsafe_allow_html=True,
    )

    st.caption("See how strongly the model separates the two classes.")

    st.markdown(
        f"**🟢 Ham — {ham_probability * 100:.1f}%**"
    )
    st.progress(ham_probability)

    st.markdown(
        f"**🔴 Spam — {spam_probability * 100:.1f}%**"
    )
    st.progress(spam_probability)


    # -----------------------------------------------------
    # Message stats
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Message insights</div>',
        unsafe_allow_html=True,
    )

    characters = len(message)
    words = len(message.split())
    exclamations = message.count("!")
    questions = message.count("?")

    c1, c2, c3, c4 = st.columns(4)

    stats = [
        (c1, characters, "Characters"),
        (c2, words, "Words"),
        (c3, exclamations, "Exclamation marks"),
        (c4, questions, "Question marks"),
    ]

    for column, value, label in stats:
        with column:
            st.markdown(
                f"""
<div class="stat">
    <div class="stat-number">{value}</div>
    <div class="stat-label">{label}</div>
</div>
""",
                unsafe_allow_html=True,
            )


    # -----------------------------------------------------
    # Technical details
    # -----------------------------------------------------

