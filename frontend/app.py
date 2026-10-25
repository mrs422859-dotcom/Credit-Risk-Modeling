"""Lauki Finance – Credit Risk Modelling (Streamlit frontend).

Talks to the FastAPI backend. Run:  streamlit run app.py
Set API_URL to point at a different backend (default http://localhost:8000).
"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Lauki Finance: Credit Risk", page_icon="📊", layout="wide")

RATING_COLORS = {"Poor": "#d64545", "Average": "#e8a33d", "Good": "#5aa469", "Excellent": "#2e8b57"}
RATING_NOTES = {
    "Poor": "High risk of default. Typically needs strong mitigants or a decline.",
    "Average": "Moderate risk. Review closely before approving.",
    "Good": "Low risk. Healthy repayment profile.",
    "Excellent": "Very low risk. Strong repayment profile.",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .rating-pill {display:inline-block; padding:.25rem .9rem; border-radius:999px;
                    color:white; font-weight:600; font-size:1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ API helpers
@st.cache_data(ttl=60, show_spinner=False)
def fetch_info():
    r = requests.get(f"{API_URL}/model/info", timeout=5)
    r.raise_for_status()
    return r.json()


def call_predict(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    if r.status_code == 422:
        errs = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in r.json()["detail"])
        raise ValueError(errs)
    r.raise_for_status()
    return r.json()


def call_batch(applications: list) -> dict:
    r = requests.post(f"{API_URL}/predict/batch", json={"applications": applications}, timeout=60)
    if r.status_code == 422:
        raise ValueError("Some rows failed validation. Check the values and column names.")
    r.raise_for_status()
    return r.json()


def gauge(score: int, rating: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 44}},
            gauge={
                "axis": {"range": [300, 900]},
                "bar": {"color": RATING_COLORS.get(rating, "#555"), "thickness": 0.35},
                "steps": [
                    {"range": [300, 500], "color": "#f6d5d5"},
                    {"range": [500, 650], "color": "#fbe9c8"},
                    {"range": [650, 750], "color": "#d8ecd9"},
                    {"range": [750, 900], "color": "#bfe3cc"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def contributions_chart(contribs: list) -> go.Figure:
    df = pd.DataFrame(contribs)
    df = df[df["value"].abs() > 1e-6].sort_values("value")
    colors = ["#d64545" if v > 0 else "#2e8b57" for v in df["value"]]
    fig = go.Figure(go.Bar(x=df["value"], y=df["label"], orientation="h", marker_color=colors))
    fig.update_layout(
        height=max(260, 32 * len(df)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Effect on log-odds of default  (red = raises risk, green = lowers)",
    )
    return fig


# ---------------------------------------------------------------------- header
st.title("📊 Lauki Finance: Credit Risk Modelling")
st.caption("Predict default probability, credit score (300–900) and rating for a loan applicant.")

try:
    info = fetch_info()
    ranges = info["training_ranges"]
except Exception:
    st.error(
        f"Cannot reach the API at **{API_URL}**. Start it with "
        "`uvicorn app.main:app --port 8000` in the `backend/` folder."
    )
    st.stop()

with st.sidebar:
    st.header("About the model")
    st.write(f"**{info['model_type']}** · {info['n_features']} features")
    m = info["metrics"]
    c1, c2 = st.columns(2)
    c1.metric("AUC", f"{m['auc']:.2f}")
    c2.metric("Gini", f"{m['gini']:.2f}")
    c1.metric("Default recall", f"{m['recall_default']:.0%}")
    c2.metric("Default precision", f"{m['precision_default']:.0%}")
    st.caption("Test-set metrics from the training notebook.")
    st.divider()
    st.subheader("Rating bands")
    for name, (lo, hi) in info["rating_bands"].items():
        st.markdown(f"<span style='color:{RATING_COLORS[name]}'>●</span> **{name}**: {lo}–{hi}", unsafe_allow_html=True)
    st.divider()
    st.caption(f"API: {API_URL}")

tab_single, tab_batch = st.tabs(["Single applicant", "Batch scoring (CSV)"])

# ---------------------------------------------------------------- single tab
with tab_single:
    with st.form("applicant"):
        st.subheader("Applicant & loan")
        c = st.columns(3)
        age = c[0].number_input("Age", 18, 100, 28, step=1)
        income = c[1].number_input("Annual income (₹)", 1, value=1_200_000, step=50_000)
        loan_amount = c[2].number_input("Loan amount (₹)", 1, value=2_560_000, step=50_000)

        c = st.columns(3)
        loan_tenure_months = c[0].number_input("Loan tenure (months)", 1, 360, 36, step=1)
        loan_purpose = c[1].selectbox("Loan purpose", ["Education", "Home", "Auto", "Personal"], index=1)
        loan_type = c[2].selectbox("Loan type", ["Unsecured", "Secured"], index=1)

        st.subheader("Credit history")
        c = st.columns(3)
        credit_utilization_ratio = c[0].slider("Credit utilization (%)", 0, 100, 30)
        delinquency_ratio = c[1].slider("Delinquency ratio (%)", 0, 100, 30)
        avg_dpd_per_delinquency = c[2].number_input("Avg DPD per delinquency", 0, value=6, step=1)

        c = st.columns(3)
        num_open_accounts = c[0].number_input("Open loan accounts", 1, 10, 2, step=1)
        residence_type = c[1].selectbox("Residence type", ["Owned", "Rented", "Mortgage"])
        lti = loan_amount / income if income else 0
        c[2].metric("Loan-to-income ratio", f"{lti:.2f}")

        submitted = st.form_submit_button("Calculate risk", type="primary", width="stretch")

    if submitted:
        payload = dict(
            age=age, income=income, loan_amount=loan_amount, loan_tenure_months=loan_tenure_months,
            avg_dpd_per_delinquency=avg_dpd_per_delinquency, delinquency_ratio=delinquency_ratio,
            credit_utilization_ratio=credit_utilization_ratio, num_open_accounts=num_open_accounts,
            residence_type=residence_type, loan_purpose=loan_purpose, loan_type=loan_type,
        )
        try:
            with st.spinner("Scoring…"):
                st.session_state["result"] = call_predict(payload)
                st.session_state["payload"] = payload
        except ValueError as e:
            st.error(f"Invalid input: {e}")
        except requests.RequestException as e:
            st.error(f"API error: {e}")

    res = st.session_state.get("result")
    if res:
        st.divider()
        for w in res["warnings"]:
            st.warning(w)

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(gauge(res["credit_score"], res["rating"]), width="stretch")
            st.markdown(
                f"<span class='rating-pill' style='background:{RATING_COLORS[res['rating']]}'>{res['rating']}</span>",
                unsafe_allow_html=True,
            )
            st.write(RATING_NOTES[res["rating"]])
        with right:
            k1, k2 = st.columns(2)
            k1.metric("Default probability", f"{res['default_probability']:.2%}")
            k2.metric("Credit score", res["credit_score"])
            st.markdown("**What drove this result**")
            st.plotly_chart(contributions_chart(res["contributions"]), width="stretch")

        st.download_button(
            "Download result (JSON)",
            data=pd.Series({**st.session_state["payload"], **{k: res[k] for k in ("default_probability", "credit_score", "rating")}}).to_json(indent=2),
            file_name="credit_risk_result.json",
            mime="application/json",
        )

# ----------------------------------------------------------------- batch tab
with tab_batch:
    cols = [
        "age", "income", "loan_amount", "loan_tenure_months", "avg_dpd_per_delinquency",
        "delinquency_ratio", "credit_utilization_ratio", "num_open_accounts",
        "residence_type", "loan_purpose", "loan_type",
    ]
    sample = pd.DataFrame(
        [
            [28, 1200000, 2560000, 36, 6, 30, 30, 2, "Owned", "Home", "Secured"],
            [45, 2800000, 1500000, 24, 0, 0, 10, 1, "Owned", "Auto", "Secured"],
            [30, 900000, 3000000, 48, 9, 60, 85, 4, "Rented", "Personal", "Unsecured"],
        ],
        columns=cols,
    )
    st.write("Upload a CSV with these columns to score many applicants at once (max 1000 rows).")
    st.download_button("Download sample CSV", sample.to_csv(index=False), "sample_applicants.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif len(df) > 1000:
            st.error("Max 1000 rows per upload.")
        else:
            try:
                out = call_batch(df[cols].to_dict("records"))
                res_df = pd.DataFrame(out["results"]).drop(columns=["index", "loan_to_income"])
                full = pd.concat([df.reset_index(drop=True), res_df], axis=1)
                st.success(f"Scored {out['count']} applicants.")
                st.dataframe(
                    full.style.format({"default_probability": "{:.2%}"}),
                    width="stretch",
                )
                dist = full["rating"].value_counts().reindex(list(RATING_COLORS)).fillna(0)
                st.bar_chart(dist)
                st.download_button("Download scored CSV", full.to_csv(index=False), "scored_applicants.csv", "text/csv")
            except ValueError as e:
                st.error(str(e))
            except requests.RequestException as e:
                st.error(f"API error: {e}")
