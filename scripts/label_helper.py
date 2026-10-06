"""
Human ground-truth labelling tool for data/labels/hand_labelled_eval.csv.

    streamlit run scripts/label_helper.py

Shows one row at a time with the keyword-rule and zero-shot suggestions side by side. Every
click is written straight back to the CSV, so the session can be stopped and resumed.
"""
import os

import pandas as pd
import streamlit as st
import yaml

HAND = "data/labels/hand_labelled_eval.csv"
ZS = "data/labels/event_labels_zeroshot.csv"

with open("config/taxonomy.yaml", "r", encoding="utf-8") as f:
    CLASSES = [c["name"] for c in yaml.safe_load(f)["classes"]]


@st.cache_data
def zero_shot_lookup() -> dict:
    if not os.path.exists(ZS):
        return {}
    zs = pd.read_csv(ZS).drop_duplicates("text")
    return dict(zip(zs["text"], zip(zs["event_type_zs"], zs["event_conf_zs"])))


def load() -> pd.DataFrame:
    df = pd.read_csv(HAND, dtype=str).fillna("")
    return df


def save(df: pd.DataFrame):
    df.to_csv(HAND, index=False)


st.set_page_config(page_title="Event ground-truth labelling", layout="centered")
st.title("Event ground-truth labelling")

df = load()
zs = zero_shot_lookup()
done = (df["user_ground_truth_event"] != "").sum()
st.progress(done / len(df), text=f"{done} / {len(df)} labelled")

if "idx" not in st.session_state:
    pending = df.index[df["user_ground_truth_event"] == ""].tolist()
    st.session_state.idx = pending[0] if pending else 0

idx = st.session_state.idx
row = df.loc[idx]
zs_label, zs_conf = zs.get(row["text"], ("", 0.0))

st.caption(f"Row {idx + 1} · {row['source']} · {row['date']} · ticker hint: {row['ticker_hint']}")
st.markdown(f"### {row['text']}")
c1, c2 = st.columns(2)
c1.metric("Keyword rule", row["rule_suggested_event"] or "-")
c2.metric("Zero-shot (BART-MNLI)", zs_label or "-", f"{float(zs_conf):.0%}" if zs_label else None,
          delta_color="off")
if row["user_ground_truth_event"]:
    st.success(f"Current label: {row['user_ground_truth_event']}")


def commit(label: str):
    df.loc[idx, "user_ground_truth_event"] = label
    df.loc[idx, "user_verified_ticker"] = st.session_state.get("ticker", row["ticker_hint"])
    save(df)
    pending = df.index[df["user_ground_truth_event"] == ""].tolist()
    st.session_state.idx = pending[0] if pending else min(idx + 1, len(df) - 1)


st.text_input("Verified ticker (blank if none / not a universe company)", value=row["user_verified_ticker"] or row["ticker_hint"], key="ticker")

b1, b2 = st.columns(2)
if zs_label and b1.button(f"Accept zero-shot: {zs_label}", use_container_width=True):
    commit(zs_label)
    st.rerun()
if row["rule_suggested_event"] and b2.button(f"Accept rule: {row['rule_suggested_event']}", use_container_width=True):
    commit(row["rule_suggested_event"])
    st.rerun()

st.write("Or choose the correct class:")
cols = st.columns(3)
for i, cls in enumerate(CLASSES):
    if cols[i % 3].button(cls, key=f"cls_{cls}", use_container_width=True):
        commit(cls)
        st.rerun()

nav1, nav2 = st.columns(2)
if nav1.button("Previous", disabled=idx == 0):
    st.session_state.idx = idx - 1
    st.rerun()
if nav2.button("Skip", disabled=idx >= len(df) - 1):
    st.session_state.idx = idx + 1
    st.rerun()
