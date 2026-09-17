"""
Energiewende Discourse Monitor — live prototype
Trains a party classifier on real GermaParl speeches, then applies it to
freshly-fetched Bundestag speeches (period 20) to predict likely party
affiliation on documents that don't carry a party label.

Secrets required (set via Streamlit Cloud -> App settings -> Secrets, NOT
committed to the repo):

    DIP_API_KEY = "your-real-key-here"
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
from datetime import date, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

st.set_page_config(page_title="Energiewende Discourse Monitor", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# Constants (from the notebook pipeline)
# ---------------------------------------------------------------------------
ENERGY_KEYWORDS = [
    "energiewende", "erneuerbare energie", "erneuerbaren energien", "windkraft",
    "windenergie", "photovoltaik", "solarenergie", "wasserstoff", "braunkohle",
    "steinkohle", "atomkraft", "kernenergie", "kohleausstieg", "atomausstieg",
    "klimaschutz", "versorgungssicherheit",
]

ENTITY_DICTIONARY = {
    "windkraft": "TECHNOLOGY", "windenergie": "TECHNOLOGY", "photovoltaik": "TECHNOLOGY",
    "solarenergie": "TECHNOLOGY", "wasserstoff": "TECHNOLOGY", "braunkohle": "TECHNOLOGY",
    "steinkohle": "TECHNOLOGY", "atomkraft": "TECHNOLOGY", "kernenergie": "TECHNOLOGY",
    "energiewende": "POLICY", "kohleausstieg": "POLICY", "atomausstieg": "POLICY",
    "klimaschutz": "POLICY", "versorgungssicherheit": "POLICY",
}

PROCEDURAL_STOPWORDS = {
    "dass", "prozent", "drucksache", "abgeordneter", "abgeordnete", "ausschuss",
    "fraktion", "beratung", "stimmen", "kollege", "kollegin", "praesident",
    "damen", "herren", "herr", "frau", "sitzung", "tagesordnung", "wort",
    "beifall", "afd", "spd", "csu", "cdu", "fdp", "bündnis", "grüne",
    "drucksach", "antrag", "absatz", "ziff", "gemäß", "abg", "grün", "bündniss",
    "linker", "präsidentin", "vizepräsidentin", "parl", "dame", "mal",
    "gemeinsam", "sehen", "liegen", "finden", "klar",
    # honorifics/address forms that leaked into top-term inspection earlier
    "geehrt", "verehrt", "präsident", "rufen", "stelle",
}

PARTY_NORMALIZE = {"LINKE": "DIE LINKE"}
PARTY_MERGE = {"CSU": "CDU/CSU", "CDU": "CDU/CSU"}
MIN_SAMPLES = 20
BASE_URL = "https://search.dip.bundestag.de/api/v1"

# ---------------------------------------------------------------------------
# spaCy (cached resource — loaded once per app instance)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading German language model…")
def load_nlp():
    import spacy
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
    nlp.max_length = 2_000_000
    return nlp

def preprocess_no_propn(nlp, text):
    doc = nlp(text.lower())
    return [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
        and token.pos_ != "PROPN"
        and token.lemma_.lower() not in PROCEDURAL_STOPWORDS
        and len(token.lemma_) > 2
    ]

def extract_entities(tokens):
    return [t for t in tokens if t in ENTITY_DICTIONARY]

# ---------------------------------------------------------------------------
# Real data: GermaParl (training data, periods 17-19)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner="Fetching GermaParl training speeches…")
def fetch_germaparl(periods=("17", "18", "19"), step=8):
    def list_session_files(period):
        url = f"https://api.github.com/repos/PolMine/GermaParlTEI/contents/{period}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return [f["name"] for f in r.json() if f["name"].endswith(".xml")]

    def download(period, filename):
        url = f"https://raw.githubusercontent.com/PolMine/GermaParlTEI/main/{period}/{filename}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text

    def parse_session_file(xml_text, period):
        root = ET.fromstring(xml_text)
        records = []
        for sp in root.iter("sp"):
            party = sp.get("party", "NA")
            text = " ".join(p.text or "" for p in sp.findall("p"))
            if any(kw in text.lower() for kw in ENERGY_KEYWORDS):
                records.append({"party": party, "text": text, "period": period})
        return records

    all_records = []
    for period in periods:
        files = list_session_files(period)[::step]
        for fname in files:
            try:
                xml_text = download(period, fname)
                all_records.extend(parse_session_file(xml_text, period))
            except Exception:
                continue
    return pd.DataFrame(all_records)

# ---------------------------------------------------------------------------
# Real data: live DIP API (period 20 — rolling recent window)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner="Fetching latest Bundestag speeches…")
def fetch_live_speeches(api_key, days_back=180, max_documents=60):
    params = {
        "apikey": api_key,
        "f.zuordnung": "BT",
        "f.datum.start": str(date.today() - timedelta(days=days_back)),
        "f.datum.end": str(date.today()),
    }
    url = f"{BASE_URL}/plenarprotokoll-text"
    all_docs = []
    while url and len(all_docs) < max_documents:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        all_docs.extend(data.get("documents", []))
        cursor = data.get("cursor")
        if not cursor:
            break
        params = {"apikey": api_key, "cursor": cursor}

    records = []
    for doc in all_docs[:max_documents]:
        text = doc.get("text", "")
        if any(kw in text.lower() for kw in ENERGY_KEYWORDS):
            records.append({
                "text": text,
                "date": doc.get("datum", "unknown"),
                "title": doc.get("titel", "Plenarprotokoll"),
            })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Train classifier (cached — retrains only when GermaParl data changes)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training party classifier on GermaParl data…")
def train_classifier(df_raw):
    nlp = load_nlp()
    df = df_raw.copy()
    df["party"] = df["party"].replace(PARTY_NORMALIZE).replace(PARTY_MERGE)
    df = df[df["party"] != "NA"]

    counts = df["party"].value_counts()
    df = df[df["party"].isin(counts[counts >= MIN_SAMPLES].index)]

    df["tokens"] = df["text"].apply(lambda t: preprocess_no_propn(nlp, t))
    df["clean_text"] = df["tokens"].apply(lambda toks: " ".join(toks))
    df["entities"] = df["tokens"].apply(extract_entities)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["party"], test_size=0.2, random_state=42, stratify=df["party"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                  stop_words=list(PROCEDURAL_STOPWORDS))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "vectorizer": vectorizer,
        "clf": clf,
        "df": df,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "y_test": y_test,
        "y_pred": y_pred,
    }

def predict_party(model, nlp, text):
    tokens = preprocess_no_propn(nlp, text)
    clean = " ".join(tokens)
    vec = model["vectorizer"].transform([clean])
    proba = model["clf"].predict_proba(vec)[0]
    classes = model["clf"].classes_
    order = np.argsort(proba)[::-1]
    return [(classes[i], proba[i]) for i in order], extract_entities(tokens)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Energiewende Discourse Monitor")
st.caption("Live German Bundestag energy-policy tracking — trained on real GermaParl speeches, applied to freshly-fetched Bundestag documents.")

if "DIP_API_KEY" not in st.secrets:
    st.error(
        "No DIP API key found. Add `DIP_API_KEY` under your app's Secrets in "
        "Streamlit Cloud (App settings → Secrets) — never commit it to the repo."
    )
    st.stop()

with st.spinner("Loading pipeline — first load can take a minute…"):
    nlp = load_nlp()
    germaparl_df = fetch_germaparl()
    model = train_classifier(germaparl_df)

tab_live, tab_classifier, tab_query, tab_recs = st.tabs(
    ["Live Feed", "Classifier Performance", "Explore by Entity", "Recommendations"]
)

# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Latest Bundestag Energy-Policy Speeches")
    col1, col2 = st.columns([1, 3])
    days_back = col1.slider("Look back (days)", 30, 365, 180, step=30)
    if col1.button("Refresh live data"):
        fetch_live_speeches.clear()

    live_df = fetch_live_speeches(st.secrets["DIP_API_KEY"], days_back=days_back)

    if live_df.empty:
        st.warning("No matching energy-related speeches found in this window. Try widening the date range.")
    else:
        rows = []
        entity_counter = {}
        for _, row in live_df.iterrows():
            predictions, entities = predict_party(model, nlp, row["text"])
            top_party, top_conf = predictions[0]
            rows.append({
                "Date": row["date"],
                "Predicted party": top_party,
                "Confidence": f"{top_conf:.0%}",
                "Entities mentioned": ", ".join(sorted(set(entities))) or "—",
                "Excerpt": row["text"][:160] + "…",
            })
            for e in entities:
                entity_counter[e] = entity_counter.get(e, 0) + 1

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if entity_counter:
            st.markdown("**Entity frequency in this window**")
            ent_df = pd.Series(entity_counter).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(ent_df.index[::-1], ent_df.values[::-1])
            plt.tight_layout()
            st.pyplot(fig)

    st.caption(
        "Predicted party is the classifier's best guess based on word choice alone — "
        "useful as a directional signal, not a substitute for verified attribution."
    )

# ---------------------------------------------------------------------------
with tab_classifier:
    st.subheader("Classifier Performance (live-trained)")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{model['accuracy']:.2f}")
    c2.metric("Macro-F1", f"{model['macro_f1']:.2f}")

    report_df = pd.DataFrame(model["report"]).T
    report_df = report_df[~report_df.index.isin(["accuracy", "macro avg", "weighted avg"])]
    st.dataframe(
        report_df[["precision", "recall", "f1-score", "support"]].round(2),
        use_container_width=True,
    )
    st.caption(
        f"Trained on {len(model['df'])} GermaParl speeches (periods 17–19), "
        f"filtered to parties with ≥{MIN_SAMPLES} samples, CDU/CSU merged as one "
        f"Fraktion, bigrams + proper-noun filtering applied."
    )

# ---------------------------------------------------------------------------
with tab_query:
    st.subheader("Explore Training Data by Entity")
    df = model["df"]
    all_entities = sorted(ENTITY_DICTIONARY.keys())
    selected = st.selectbox("Entity", all_entities)
    matches = df[df["entities"].apply(lambda ents: selected in ents)]

    st.write(f"**{len(matches)}** training speeches mention *{selected}*")
    if not matches.empty:
        st.dataframe(
            matches[["party", "period"]].value_counts().reset_index(name="count"),
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------------------
with tab_recs:
    st.subheader("How to Read This Tool")
    st.markdown("""
- **Predicted party ≠ confirmed party.** It's a lexical-similarity signal from a
  classifier trained on real historical speeches — treat low-confidence predictions
  (under ~40%) as inconclusive, not wrong.
- **Consistency across time is the strongest signal.** A topic or framing that shows
  up across multiple refreshes and multiple parties is more durable than a single
  speech.
- **CDU and CSU are tracked as one bloc**, since they vote together as a joint
  Fraktion — their combined signal shouldn't be read as two independent endorsements.
- **This is a prototype, not a finished product.** It has no user accounts, no
  scheduled background refresh (data updates only when you open the app or click
  Refresh), and results are cached for performance — always check the "Look back"
  window matches what you intend to analyze.
""")

st.divider()
st.caption("M508 Big Data Analytics — Energiewende Discourse Analysis · Live prototype, not production-hardened.")
