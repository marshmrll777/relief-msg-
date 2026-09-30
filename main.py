from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import re
import nltk
import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

nltk.download("stopwords")

app = FastAPI(
    title="ReliefAI NLP API"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Request Model
# =========================

class MessageRequest(BaseModel):
    messages: list[str]
    n_clusters: int = 5


# =========================
# Text Cleaning
# =========================

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =========================
# Cluster Names
# =========================

def detect_cluster_name(messages):

    text = " ".join(messages).lower()

    if any(word in text for word in [
        "food",
        "hungry",
        "rice",
        "water",
        "meal"
    ]):
        return "Food Requests"

    if any(word in text for word in [
        "doctor",
        "medical",
        "medicine",
        "ambulance",
        "treatment"
    ]):
        return "Medical Requests"

    if any(word in text for word in [
        "shelter",
        "tent",
        "home"
    ]):
        return "Shelter Requests"

    if any(word in text for word in [
        "rescue",
        "trapped",
        "evacuation",
        "boat"
    ]):
        return "Rescue Requests"

    return "Others"


# =========================
# Health Check
# =========================

@app.get("/")
def home():

    return {
        "status": "ReliefAI NLP API Running"
    }


# =========================
# Clustering Endpoint
# =========================

@app.post("/cluster")
def cluster_messages(request: MessageRequest):

    messages = request.messages

    cleaned = [
        clean_text(msg)
        for msg in messages
    ]

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(cleaned)

    n_clusters = min(
        request.n_clusters,
        len(messages)
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # =========================
    # Cluster Summary
    # =========================

    cluster_data = {}

    for i, label in enumerate(labels):

        label = int(label)

        if label not in cluster_data:
            cluster_data[label] = []

        cluster_data[label].append(messages[i])

    total_messages = len(messages)

    cluster_summary = []

    for cluster_id, msgs in cluster_data.items():

        percentage = round(
            len(msgs) * 100 / total_messages,
            2
        )

        cluster_summary.append({
            "cluster_id": cluster_id,
            "cluster_name": detect_cluster_name(msgs),
            "count": len(msgs),
            "percentage": percentage,
            "sample_messages": msgs[:3]
        })

    # =========================
    # t-SNE Visualization
    # =========================

    if len(messages) > 2:

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(
                5,
                len(messages)-1
            )
        )

        points = tsne.fit_transform(
            X.toarray()
        )

    else:

        points = np.zeros(
            (len(messages),2)
        )

    visualization = []

    for i in range(len(messages)):

        visualization.append({
            "message": messages[i],
            "cluster": int(labels[i]),
            "x": float(points[i][0]),
            "y": float(points[i][1])
        })

    return {
        "total_messages": total_messages,
        "cluster_summary": cluster_summary,
        "visualization": visualization
    }