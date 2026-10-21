"""IncomeLens - income-bracket prediction from census profile (Streamlit).
Run: streamlit run app.py
Model: XGBoost on UCI Adult (1994 US census, 30,162 cleaned rows).
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="IncomeLens - Salary Prediction", page_icon="💼", layout="wide")

@st.cache_resource
def load_model():
    m = XGBClassifier(); m.load_model(os.path.join(HERE,"artifacts","model.json"))
    return m

@st.cache_data
def load_meta():
    a = os.path.join(HERE,"artifacts")
    cats = json.load(open(os.path.join(a,"categories.json")))
    metrics = json.load(open(os.path.join(a,"metrics.json")))
    brk = pd.read_csv(os.path.join(a,"breakdowns.csv"))
    return cats, metrics, brk

model = load_model()
meta, metrics, brk = load_meta()
FEATS, CATS = meta["features"], meta["categories"]

st.title("💼 IncomeLens")
st.caption("Which profiles earn above $50K? XGBoost trained on 30,162 real census records "
           "(UCI Adult dataset).")

with st.sidebar:
    st.header("Your profile")
    age = st.slider("Age", 17, 90, 35)
    edu = st.selectbox("Education", CATS["education"], index=CATS["education"].index("Bachelors"))
    hours = st.slider("Hours worked per week", 1, 99, 40)
    occ = st.selectbox("Occupation", CATS["occupation"])
    work = st.selectbox("Work sector", CATS["workclass"])
    mar = st.selectbox("Marital status", CATS["marital_status"])
    rel = st.selectbox("Relationship", CATS["relationship"])
    race = st.selectbox("Race", CATS["race"])
    sex = st.selectbox("Sex", CATS["sex"])
    gain = st.number_input("Yearly capital gains ($)", 0, 100000, 0, 500)
    loss = st.number_input("Yearly capital losses ($)", 0, 5000, 0, 100)
    st.divider()
    st.subheader("Honest limits")
    st.caption(f"{metrics['accuracy']*100:.0f}% accuracy, AUC {metrics['roc_auc']} on holdout. "
               "The data is the 1994 US census - $50K then is roughly $105K+ now, and "
               "patterns reflect that era and country. A teaching model, not career advice.")

EDU_NUM = {"Preschool":1,"1st-4th":2,"5th-6th":3,"7th-8th":4,"9th":5,"10th":6,"11th":7,
           "12th":8,"HS-grad":9,"Some-college":10,"Assoc-voc":11,"Assoc-acdm":12,
           "Bachelors":13,"Masters":14,"Prof-school":15,"Doctorate":16}

def predict(profile):
    row = {f: 0 for f in FEATS}
    for k, v in profile["cats"].items():
        col = f"{k}_{v}"
        if col in row: row[col] = 1
    row.update(profile["nums"])
    return float(model.predict_proba(pd.DataFrame([row])[FEATS])[0,1])

profile = {"cats": {"workclass":work,"education":edu,"marital_status":mar,
                    "occupation":occ,"relationship":rel,"race":race,"sex":sex},
           "nums": {"age":age,"education_num":EDU_NUM[edu],"capital_gain":gain,
                    "capital_loss":loss,"hours_per_week":hours}}
p = predict(profile)

c1, c2 = st.columns(2)
c1.metric("P(income above $50K)", f"{p*100:.0f}%")
c1.progress(min(1.0, p))
c2.metric("Dataset baseline", "24%", "share above $50K in census data")

st.subheader("How each factor shifts the odds (census reality)")
brk_show = brk[brk["count"] >= 100].copy()
brk_show["pct"] = (brk_show["mean"]*100).round(1)
pick = st.selectbox("Slice by", sorted(brk_show.feature.unique()))
d = brk_show[brk_show.feature == pick].sort_values("pct", ascending=False)
st.bar_chart(d.set_index("value")["pct"])

st.subheader("What the model weighs most")
imp = pd.DataFrame({"feature": FEATS, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False).head(12)
imp["feature"] = imp.feature.str.replace("_", " ").str.replace("capital gain", "capital gains")
st.bar_chart(imp.set_index("feature"))
