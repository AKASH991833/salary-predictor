"""Train IncomeLens on the UCI Adult census dataset (32,561 people, 1994 US census).
Model: gradient-boosted trees predicting P(income > $50K) from profile features.
Artifacts: model.json, metrics.json, categories.json, breakdowns.csv
Run: python train.py
"""
import json, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ["workclass","education","marital_status","occupation","relationship","race","sex"]
NUMS = ["age","education_num","capital_gain","capital_loss","hours_per_week"]

def load():
    cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
            "occupation","relationship","race","sex","capital_gain","capital_loss",
            "hours_per_week","native_country","income"]
    df = pd.read_csv(os.path.join(HERE,"data","adult.data"), header=None,
                     names=cols, skipinitialspace=True)
    df = df[(df != "?").all(axis=1)].copy()          # drop unknown-category rows
    df["y"] = (df.income == ">50K").astype(int)
    return df

def main():
    df = load()
    X = pd.get_dummies(df[CATS + NUMS], columns=CATS)
    feats = X.columns.tolist()
    cut = int(len(df)*0.8)                            # shuffled? keep random split instead
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, df.y, test_size=0.2, random_state=42, stratify=df.y)
    model = XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.06,
                          subsample=0.9, colsample_bytree=0.8, n_jobs=4, random_state=42)
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:,1]
    acc = float(accuracy_score(yte, prob > 0.5)); auc = float(roc_auc_score(yte, prob))
    print(f"rows={len(df)} acc={acc:.3f} AUC={auc:.3f}")

    out = os.path.join(HERE,"artifacts"); os.makedirs(out, exist_ok=True)
    model.save_model(os.path.join(out,"model.json"))
    cats = {c: sorted(df[c].unique().tolist()) for c in CATS}
    json.dump({"features":feats,"categories":cats},
              open(os.path.join(out,"categories.json"),"w"), indent=2)
    json.dump({"accuracy":round(acc,3),"roc_auc":round(auc,3),"n_rows":int(len(df)),
               "n_test":int(len(yte)),"dataset":"UCI Adult (1994 US census extract)",
               "target":"P(annual income > $50K)"},
              open(os.path.join(out,"metrics.json"),"w"), indent=2)
    rows = []
    for c in CATS:
        g = df.groupby(c).y.agg(["mean","count"]).reset_index()
        g["feature"] = c; g = g.rename(columns={c:"value"})
        rows.append(g[["feature","value","mean","count"]])
    pd.concat(rows).to_csv(os.path.join(out,"breakdowns.csv"), index=False)
    print("artifacts written")

if __name__ == "__main__":
    main()
