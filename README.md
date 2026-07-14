# Employee Attrition Analytics — Streamlit App

A capstone-ready data analytics project: EDA + key-driver analysis + a predictive
(Random Forest) model + recommendations for employee attrition, built on the public
**IBM HR Analytics Employee Attrition** dataset (1,470 employees, 35 features).

## Project structure
```
hr-attrition-app/
├── app.py                          # Streamlit app (5 pages)
├── requirements.txt
└── data/
    └── HR-Employee-Attrition.csv   # dataset
```

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Get a shareable public link (free, ~5 minutes) — Streamlit Community Cloud
1. Create a free GitHub account if you don't have one, and a new **public** repo
   (e.g. `hr-attrition-app`).
2. Upload the 3 items in this folder (`app.py`, `requirements.txt`, `data/` folder)
   to that repo — either via the GitHub web UI ("Add file → Upload files") or:
   ```bash
   git init
   git add .
   git commit -m "HR attrition analytics app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/hr-attrition-app.git
   git push -u origin main
   ```
3. Go to **https://share.streamlit.io** → sign in with GitHub → **New app**.
4. Select your repo, branch `main`, main file path `app.py` → **Deploy**.
5. In 1–2 minutes you'll get a public link like:
   `https://<your-username>-hr-attrition-app.streamlit.app`
   — that's your presentable link.

## What's in the app
- **Business Problem** — framing, headline KPIs, cost-of-attrition estimate
- **Exploratory Analysis** — attrition split, department/age/income/job-role breakdowns, correlation heatmap
- **Key Drivers of Attrition** — overtime, work-life balance, tenure, promotions, commute, travel
- **Predictive Model** — Random Forest classifier (ROC-AUC ~0.78), confusion matrix, feature
  importance, and an interactive "what-if" risk calculator
- **Recommendations** — prioritized, data-backed HR actions

All charts respond to the sidebar filters (Department, Gender, Age range).

## Notes for your presentation
- Model AUC ≈ 0.78 on held-out test data — solid for a 1,470-row dataset with an
  imbalanced target (~16% attrition rate); mention this as a strength/limitation.
- Talk track: Problem → EDA findings → statistically strongest drivers → model that
  operationalizes those drivers → recommendations tied back to the drivers.
