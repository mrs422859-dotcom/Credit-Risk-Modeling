import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient

from app.main import app

GOOD = dict(age=28, income=1200000, loan_amount=2560000, loan_tenure_months=36,
            avg_dpd_per_delinquency=6, delinquency_ratio=30, credit_utilization_ratio=30,
            num_open_accounts=2, residence_type="Owned", loan_purpose="Home", loan_type="Secured")


def test_all():
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"]
        assert c.get("/model/info").json()["n_features"] == 13
        r = c.post("/predict", json=GOOD)
        assert r.status_code == 200, r.text
        j = r.json()
        assert 300 <= j["credit_score"] <= 900 and 0 <= j["default_probability"] <= 1
        assert j["warnings"] == []
        # score formula consistency
        assert j["credit_score"] == int(300 + (1 - j["default_probability"]) * 600)
        # original UI default (avg DPD 20) is outside training range -> warning
        assert c.post("/predict", json={**GOOD, "avg_dpd_per_delinquency": 20}).json()["warnings"]
        # validation
        assert c.post("/predict", json={**GOOD, "age": 5}).status_code == 422
        assert c.post("/predict", json={**GOOD, "income": 0}).status_code == 422
        assert c.post("/predict", json={**GOOD, "loan_type": "x"}).status_code == 422
        # batch matches single
        b = c.post("/predict/batch", json={"applications": [GOOD, GOOD]}).json()
        assert b["count"] == 2 and b["results"][0]["credit_score"] == j["credit_score"]
