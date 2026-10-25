# Lauki Finance – Credit Risk (FastAPI + Streamlit)

```
backend/   FastAPI service (model loaded once, /docs auto-generated)
frontend/  Streamlit UI that calls the API
```

## Run
```bash
# terminal 1
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000     # docs: http://localhost:8000/docs

# terminal 2
cd frontend && pip install -r requirements.txt
streamlit run app.py                          # set API_URL if the API is elsewhere
```
Tests: `cd backend && pytest`

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness + model loaded |
| GET | /model/info | features, training ranges, rating bands, notebook metrics |
| POST | /predict | one applicant -> probability, score, rating, feature contributions, range warnings |
| POST | /predict/batch | up to 1000 applicants |

Scoring maths is identical to the original `prediction_helper.py`
(score = 300 + (1 - p) * 600; Poor <500, Average <650, Good <750, Excellent <=900).
