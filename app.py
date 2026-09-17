import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Energiewende Discourse Monitor", page_icon="🗳️", layout="wide")

# ---------------------------------------------------------------------------
# Data (project results — replace with real df/entities export when available)
# ---------------------------------------------------------------------------
PARTIES = ["AfD", "CDU/CSU", "DIE LINKE", "FDP", "GRUENE", "SPD"]

f1_data = pd.DataFrame({
    "Party": ["CDU/CSU", "AfD", "GRUENE", "SPD", "FDP", "DIE LINKE"],
    "F1-score": [0.62, 0.50, 0.50, 0.49, 0.27, 0.24],
    "Training examples": [385, 43, 240, 299, 84, 107],
})

confusion = pd.DataFrame(
    [[5, 2, 0, 0, 2, 0],
     [1, 47, 7, 6, 7, 9],
     [1, 3, 7, 0, 7, 3],
     [1, 8, 2, 4, 2, 0],
     [1, 3, 6, 1, 26, 11],
     [2, 13, 2, 1, 13, 29]],
    index=PARTIES, columns=PARTIES
)

top_terms = {
    "AfD": ["sogenannter", "wasserstoff", "bürger", "gebiet", "kriegsverbrechen",
            "landwirte", "sparer", "geehrt", "präsident", "erzeugen"],
    "CDU/CSU": ["kernenergie", "unser", "land", "wichtig", "europäisch", "sicherheit",
                "bereich", "vereinbarung", "gelten", "besonderer"],
    "DIE LINKE": ["geld", "kosten", "linke", "bahn", "atomkraft", "bürgerenergie",
                  "bericht", "interessieren", "süden", "brennelement"],
    "FDP": ["koalition", "frei demokrat", "demokrat", "märz", "verehrt", "bundesrat",
            "verbraucher", "opposition", "seite", "planwirtschaftlich"],
    "GRUENE": ["atomkraftwerk", "eigentlich", "fragen", "minister", "klimaschutz",
               "abkommen", "energiewend", "antwort", "wirtschaftsminister", "defizit"],
    "SPD": ["insofern", "lieb kolleginn", "glauben", "wirtschaftlich", "handwerk",
            "kolleginn", "atomenergie", "wichtig", "verlängerung", "klage"],
}

evaluation = pd.DataFrame({
    "Component": ["LDA", "BERTopic", "Party classifier", "Party classifier"],
    "Metric": ["c_v coherence", "c_v coherence", "Accuracy", "Macro-F1"],
    "Score": [0.374, 0.397, 0.496, 0.434],
})

entity_pool = {
    "AfD": ["Wasserstoff", "Landwirte", "Sparer"],
    "CDU/CSU": ["Kernenergie", "Energieversorgung", "Sicherheit"],
    "DIE LINKE": ["Atomkraft", "Bürgerenergie", "Klimawandel"],
    "FDP": ["Verbraucher", "Koalition", "Wettbewerb"],
    "GRUENE": ["Atomkraftwerk", "Klimaschutz", "Energiewende"],
    "SPD": ["Atomenergie", "Verlängerung", "Handwerk"],
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🗳️ Energiewende Discourse Monitor")
st.caption(
    "German Bundestag energy-policy rhetoric — topic modeling, party classification, "
    "and live query demo. M508 Big Data Analytics NLP project."
)

tab_overview, tab_classifier, tab_query, tab_eval, tab_recs = st.tabs(
    ["Overview", "Party Classifier", "Live Query", "Evaluation", "Recommendations"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Project Summary")
    st.write(
        "Tracking how German Bundestag energy-policy language shifted before, during, "
        "and after the 2022 energy crisis, using topic modeling (LDA → BERTopic), "
        "dictionary-based NER, and a supervised party classifier as a lexical-"
        "differentiation check."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Periods covered", "2009–2025")
    c2.metric("Energy-related speeches", "1,168")
    c3.metric("Data sources", "2")
    c4.metric("LDA topics fit", "8")

    st.subheader("Key Finding")
    st.info(
        "Climate-transition language (\"Klimaschutz,\" \"Energiewende\") remained the "
        "dominant frame **even through the 2022 security-of-supply crisis**, rather "
        "than being displaced by crisis-specific vocabulary — evidence that political "
        "rhetoric on this topic shows more continuity than reactivity."
    )

# ---------------------------------------------------------------------------
# Party Classifier
# ---------------------------------------------------------------------------
with tab_classifier:
    st.subheader("Party Classification Results")
    st.write(
        "TF-IDF + Logistic Regression, predicting speaker party from speech text "
        "alone. CDU/CSU merged (joint Fraktion); bigrams and proper-noun filtering "
        "applied."
    )
    c1, c2 = st.columns(2)
    c1.metric("Accuracy (6-class, chance = 0.17)", "0.50")
    c2.metric("Macro-F1", "0.43")

    st.markdown("**Per-Party Performance**")
    st.dataframe(f1_data, use_container_width=True, hide_index=True)

    st.markdown("**Confusion Matrix** (rows = true party, columns = predicted)")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(PARTIES))); ax.set_xticklabels(PARTIES, rotation=45, ha="right")
    ax.set_yticks(range(len(PARTIES))); ax.set_yticklabels(PARTIES)
    for i in range(len(PARTIES)):
        for j in range(len(PARTIES)):
            v = confusion.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > confusion.values.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Top Distinctive Terms per Party**")
    for p in PARTIES:
        st.markdown(f"*{p}:* " + ", ".join(top_terms[p]))

# ---------------------------------------------------------------------------
# Live Query
# ---------------------------------------------------------------------------
with tab_query:
    st.subheader("Live Discourse Query")
    st.write(
        "Filter by party and/or legislative period to see which energy entities "
        "dominated that slice of discourse. **Demo data** — mirrors the notebook's "
        "`query_discourse()` function; swap in the real `entities`/`party`/`period` "
        "export to make this live against actual data."
    )
    col1, col2 = st.columns(2)
    party_sel = col1.selectbox("Party", ["All"] + PARTIES)
    period_sel = col2.selectbox("Period", ["All", "17", "18", "19", "20"])

    active_parties = PARTIES if party_sel == "All" else [party_sel]
    rng = np.random.default_rng(abs(hash((party_sel, period_sel))) % (2**32))
    n_speeches = int(20 + rng.integers(0, 180))

    counts = {}
    for p in active_parties:
        for e in entity_pool[p]:
            counts[e] = counts.get(e, 0) + int(3 + rng.integers(0, 15))
    result_df = (
        pd.DataFrame(counts.items(), columns=["Entity", "Mentions (relative)"])
        .sort_values("Mentions (relative)", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    st.caption(f"n≈{n_speeches} speeches | party={party_sel} | period={period_sel}")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation Summary")
    st.dataframe(evaluation, use_container_width=True, hide_index=True)

    st.markdown("**Topic modeling:** BERTopic outperforms LDA on both coherence and "
                "qualitative readability, cleanly separating nuclear power, climate "
                "targets, and renewable-energy sub-themes that LDA merged into one cluster.")
    st.markdown("**Classifier:** Reaches ~3x chance accuracy, with errors clustering "
                "along real political structure — CDU/CSU (joint Fraktion) and "
                "GRUENE/SPD (coalition-era overlap) — rather than appearing random.")
    st.markdown("**NER spot-check:** All 5 sampled entity matches were contextually "
                "correct, confirming the dictionary-based tagging works as intended.")

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
with tab_recs:
    st.subheader("Business Recommendations")
    st.markdown("""
- Don't take a single party's statement at face value — the tool checks whether the same language shows up across multiple parties over multiple years. Consistent cross-party language signals a durable commitment; single-party, single-period language signals exposure risk.
- The tool treats CDU and CSU as one bloc (not two), since they vote together — so their combined rhetoric isn't double-counted as broader political support.
- The tool flags rhetoric that survived the 2022 crisis unchanged (like climate framing did) as a stronger signal of what will hold going forward.
- Before a long-term investment decision (hydrogen infrastructure, renewables), the tool surfaces whether current rhetoric is tied to the sitting coalition — so users know if it's likely to shift after a coalition change.
- The tool sends alerts when a party's energy-policy language shifts significantly period-over-period, so users catch a rhetoric change before it becomes a policy reversal, not after.
- New Bundestag sessions get ingested automatically as they're published, so these signals stay current rather than based on a one-time historical snapshot.
""")

st.divider()
st.caption("Built for M508 Big Data Analytics — Energiewende Discourse Analysis · Demo dashboard, not a production system")
