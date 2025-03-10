import os
import re
import string
import multiprocessing
import warnings
import numpy as np
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from joblib import Parallel, delayed, dump, load
import streamlit as st

# Set page configuration for the Streamlit app
st.set_page_config(page_title="IMDB Movie Review", layout="wide", page_icon=":label:")

warnings.filterwarnings("ignore")

# Download required NLTK data quietly if not already present
for pkg in ['stopwords', 'punkt', 'wordnet']:
    nltk.download(pkg, quiet=True)

# Preload resources for efficiency
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

def add_custom_css():
    """Inject custom CSS to style the Streamlit app."""
    st.markdown(
        """
        <style>
        h1 {
            color: #ff4b4b !important; /* Title in a vibrant red */
            font-size: 3em !important;
            font-weight: bold !important;
        }
        h2 {
            color: #1a73e8 !important; /* Headers in blue */
            font-size: 2.5em !important;
            font-weight: bold !important;
        }
        h3 {
            color: #4285f4 !important; /* Subheaders in lighter blue */
            font-size: 2em !important;
            font-weight: bold !important;
        }
        p {
            color: #333333 !important; /* Paragraph text dark gray */
            font-size: 1.2em !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

def preprocess_text(text, use_stemming=False, remove_digits=True):
    """Clean and preprocess text with optional stemming and digit removal."""
    text = text.lower()
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in STOPWORDS]
    if use_stemming:
        words = [STEMMER.stem(word) for word in words]
    else:
        words = [LEMMATIZER.lemmatize(word) for word in words]
    return " ".join(words)

def parallel_preprocess(text_series, n_jobs=2, use_stemming=False, remove_digits=True):
    """Apply text preprocessing in parallel."""
    processed = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_text)(text, use_stemming, remove_digits) for text in text_series
    )
    return processed

@st.cache_data(show_spinner=False)
def load_data(n_jobs=2, use_stemming=False, remove_digits=True):
    """
    Load and preprocess the dataset using parallel processing.
    A cached CSV file is created for given preprocessing options.
    """
    cache_file = f"IMDB_Dataset_Cleaned_stemming_{use_stemming}_digits_{remove_digits}.csv"
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file)
    else:
        data = pd.read_csv("IMDB_Dataset.csv")
        data = data.sample(10000, random_state=42)  # Subsample for performance
        st.info("Preprocessing data in parallel. This may take a moment...")
        data['cleaned_text'] = parallel_preprocess(
            data['review'], n_jobs=n_jobs, use_stemming=use_stemming, remove_digits=remove_digits
        )
        data.to_csv(cache_file, index=False)
    data = data.sample(10000, random_state=42)
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    data['review_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
    return data

@st.cache_resource(show_spinner=False)
def vectorize_text(corpus, max_features=3000, ngram_range=(1,1)):
    """Vectorize text using TF-IDF and cache the vectorizer."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, binary=True)
    X_vec = vectorizer.fit_transform(corpus).astype('float32')
    return vectorizer, X_vec

# Note the underscore before _X_train_vec to prevent Streamlit from trying to hash a sparse matrix.
@st.cache_resource(show_spinner=False)
def train_models(_X_train_vec, y_train):
    """Train classifiers and return the trained models."""
    models = {}
    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    # Voting ensemble: aggregates predictions from all classifiers
    ensemble = VotingClassifier(estimators=[('nb', nb), ('lr', lr), ('rf', rf)], voting='hard')
    
    models["Naïve Bayes"] = nb.fit(_X_train_vec, y_train)
    models["Logistic Regression"] = lr.fit(_X_train_vec, y_train)
    models["Random Forest"] = rf.fit(_X_train_vec, y_train)
    models["Ensemble"] = ensemble.fit(_X_train_vec, y_train)
    
    # Persist models to disk for future use
    for name, model in models.items():
        dump(model, f"{name}.joblib")
    return models

def evaluate_models(models, X_test_vec, y_test):
    """Evaluate each model and return performance metrics."""
    performance = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        # Compute ROC curve and AUC if available
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_vec)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
        performance[name] = {
            "accuracy": acc, 
            "report": report, 
            "confusion_matrix": cm, 
            "fpr": fpr, 
            "tpr": tpr, 
            "roc_auc": roc_auc
        }
    return performance

def plot_confusion_matrix(cm, title):
    """Plot and return a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_roc_curve(fpr, tpr, roc_auc, title):
    """Plot and return an ROC curve figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    return fig

def plot_review_length_distribution(data):
    """Plot and return a histogram of review lengths."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['review_length'], bins=30, kde=True, ax=ax)
    ax.set_title("Review Length Distribution")
    ax.set_xlabel("Number of Words")
    return fig

def word_frequency_by_sentiment(data, sentiment_value):
    """Return a frequency distribution of words for a given sentiment."""
    from collections import Counter
    reviews = data[data['sentiment'] == sentiment_value]['cleaned_text']
    words = " ".join(reviews).split()
    freq = Counter(words)
    return freq.most_common(20)

def sentiment_dashboard():
    add_custom_css()  # Apply custom styles at the beginning
    st.title("Advanced IMDB Movie Review Sentiment Analysis")
    
    # Sidebar configuration for preprocessing and vectorization settings
    st.sidebar.header("Settings")
    n_jobs = st.sidebar.slider("Number of CPU cores for preprocessing", 1, multiprocessing.cpu_count(), 2)
    use_stemming = st.sidebar.checkbox("Use Stemming Instead of Lemmatization", value=False)
    remove_digits = st.sidebar.checkbox("Remove Digits", value=True)
    max_features = st.sidebar.slider("Max Features for TF-IDF", 1000, 5000, 3000, step=500)
    ngram_option = st.sidebar.selectbox("Select n-gram range", ["Unigram", "Bigram", "Unigram + Bigram"])
    if ngram_option == "Unigram":
        ngram_range = (1, 1)
    elif ngram_option == "Bigram":
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 2)
    
    selected_model = st.sidebar.selectbox("Select Classifier", 
                                            ["Logistic Regression", "Naïve Bayes", "Random Forest", "Ensemble"])
    
    # Load and preprocess data
    data = load_data(n_jobs=n_jobs, use_stemming=use_stemming, remove_digits=remove_digits)
    st.subheader("Data Sample")
    st.write(data.head())
    
    # Vectorize the entire cleaned corpus with current settings
    vectorizer, _ = vectorize_text(data['cleaned_text'], max_features=max_features, ngram_range=ngram_range)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], data['sentiment'], test_size=0.2, random_state=42
    )
    X_train_vec = vectorizer.transform(X_train).astype('float32')
    X_test_vec = vectorizer.transform(X_test).astype('float32')
    
    # Train or load models
    try:
        models = {name: load(f"{name}.joblib") for name in ["Naïve Bayes", "Logistic Regression", "Random Forest", "Ensemble"]}
    except Exception as e:
        st.warning("Pre-trained models not found or failed to load. Training models now...")
        models = train_models(X_train_vec, y_train)
    
    # Evaluate all models
    performance = evaluate_models(models, X_test_vec, y_test)
    st.sidebar.subheader("Model Performance")
    for name, metrics in performance.items():
        st.sidebar.write(f"{name} Accuracy: {metrics['accuracy']:.2f}")
    
    # Main Dashboard Visualizations
    st.subheader("Sentiment Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.countplot(data=data, x='sentiment', ax=ax1)
    st.pyplot(fig1)
    
    st.subheader("Review Length Distribution")
    fig_length = plot_review_length_distribution(data)
    st.pyplot(fig_length)
    
    st.subheader("Word Cloud for Movie Reviews")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['cleaned_text']))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis("off")
    st.pyplot(fig2)
    
    st.subheader("Word Frequency Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Most Frequent Words in Positive Reviews")
        pos_freq = pd.DataFrame(word_frequency_by_sentiment(data, 1), columns=["Word", "Frequency"])
        st.table(pos_freq)
    with col2:
        st.markdown("### Most Frequent Words in Negative Reviews")
        neg_freq = pd.DataFrame(word_frequency_by_sentiment(data, 0), columns=["Word", "Frequency"])
        st.table(neg_freq)
    
    # Prediction Section for new reviews
    st.subheader("Predict Sentiment of a New Review")
    user_input = st.text_area("Enter movie review:")
    if user_input:
        user_input_cleaned = preprocess_text(user_input, use_stemming=use_stemming, remove_digits=remove_digits)
        user_vec = vectorizer.transform([user_input_cleaned])
        prediction = models[selected_model].predict(user_vec)[0]
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        st.write(f"Predicted Sentiment using {selected_model}: {sentiment_label}")
    
    # Detailed Performance Visualization for the selected model
    st.subheader(f"{selected_model} Performance Details")
    cm_fig = plot_confusion_matrix(performance[selected_model]["confusion_matrix"], f"{selected_model} Confusion Matrix")
    st.pyplot(cm_fig)
    
    if performance[selected_model]["fpr"] is not None:
        roc_fig = plot_roc_curve(
            performance[selected_model]["fpr"], 
            performance[selected_model]["tpr"], 
            performance[selected_model]["roc_auc"], 
            f"{selected_model} ROC Curve"
        )
        st.pyplot(roc_fig)
    
    with st.expander("View Detailed Classification Report"):
        report_df = pd.DataFrame(performance[selected_model]["report"]).transpose()
        st.table(report_df)
    
    st.info("Advanced analysis complete. Models and preprocessing steps are cached for faster subsequent runs.")

if __name__ == "__main__":
    sentiment_dashboard()
