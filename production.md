## Production Deployment Guide (Render)

This guide explains how to push code and deploy the API on Render.

### 1) Prerequisites
- A GitHub or GitLab repository containing this project
- A Render account
- Python version: 3.11.x

### 2) Local preparation
- Models and data are large; they are ignored by `.gitignore`. Ensure trained artifacts are available in the repo if you want them deployed, or upload to object storage and adjust loading accordingly.
- Do NOT commit any `.env` files. Secrets will be set in Render.

### 3) Repository setup
- Ensure these files exist at repo root:
  - `render.yaml`
  - `Requirements.txt` (Render uses this name here)
  - `main.py`
  - `.gitignore`

### 4) Push to Git
```bash
# initialize (if not already a repo)
git init

git add .
# first commit
git commit -m "Initial commit: Loan API with Render config"

# add remote (replace with your repo URL)
git remote add origin <your-repo-url>

git push -u origin main  # or master
```

### 5) Render deployment via render.yaml
Render will autodetect `render.yaml` when you create a new Blueprint.

- On Render dashboard:
  1. New → Blueprint
  2. Connect your repo
  3. Render will read `render.yaml` and provision:
     - A Python Web Service named `loan-api`
     - A managed Postgres database `loan-db`
  4. Click Apply

The web service uses:
- Build: `pip install -r Requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Env vars:
  - `DATABASE_URL` injected from `loan-db`
  - `USE_SQLITE=false`

### 6) Environment variables
- In Render → loan-api → Environment:
  - Confirm `DATABASE_URL` exists (injected by the database resource)
  - Add `USE_SQLITE=false` to force Postgres in production

### 7) Database initialization and data loading
- On first boot, the service auto-creates the `cibil_scores` table (Postgres) via `init_postgres_db()`.
- The app attempts to check data presence and will load `cibil_database.csv` into the DB if empty.
  - Ensure `cibil_database.csv` is present in the repo if you want auto-seeding.

### 8) Health checks
- After deploy completes, open:
  - `GET /` → basic status
  - `GET /health` → detailed readiness (models/encoders/db flags)

### 9) Subsequent updates
```bash
git add -A
git commit -m "Deploy: <short description>"
git push
```
Render will rebuild and redeploy automatically from your default branch.

### 10) Troubleshooting
- Service fails booting with model file not found:
  - Ensure `eligibility_model.pkl`, `product_model.pkl`, `amount_model.pkl`, `tenure_model.pkl`, `rate_model.pkl`, and `label_encoders.pkl` are present in the repo or adjust code to fetch them from storage.
- Database errors:
  - Verify `DATABASE_URL` is present and `USE_SQLITE=false`.
- CSV seeding:
  - Make sure `cibil_database.csv` exists and matches expected columns: `CIBIL ID`, `CIBIL Score`.

### 11) Security notes
- Never commit secrets or `.env` files.
- Use Render Secrets for sensitive values.

### 12) Rollback
- In Render, select a previous successful deploy and roll back.
