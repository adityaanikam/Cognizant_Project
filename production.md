## Production Deployment Guide (Render)

This guide explains how to push code and deploy the API on Render.

### 1) Prerequisites
- A GitHub or GitLab repository containing this project
- A Render account
- Python version: 3.11.x (compatible with all packages)

### 2) Local preparation
- Models and data files are now included in the repo for deployment
- Do NOT commit any `.env` files. Secrets will be set in Render.
- Ensure all ML model files (*.pkl) and CSV data are present

### 3) Repository setup
- Ensure these files exist at repo root:
  - `render.yaml` ✅
  - `Requirements.txt` ✅
  - `main.py` ✅
  - `database.py` ✅
  - `runtime.txt` ✅ (specifies Python 3.11.4)
  - `.gitignore` ✅
  - All ML model files (*.pkl) ✅
  - `cibil_database.csv` ✅

### 4) Push to Git
```bash
# initialize (if not already a repo)
git init

git add .
# first commit
git commit -m "Initial commit: Loan API with Render config and Python 3.11 compatibility"

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
     - A Python Web Service named `loan-api` (Python 3.11)
     - A managed Postgres database `loan-db`
  4. Click Apply

The web service uses:
- Build: `pip install -r Requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Python: 3.11.4 (compatible with all packages)
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
- Retry logic handles database connection delays during startup.

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
- **Build fails with Python version error**: Ensure `runtime.txt` specifies `python-3.11.4`
- **Package compatibility errors**: All packages in `Requirements.txt` are tested with Python 3.11
- **Service fails booting with model file not found**: Ensure all *.pkl files are in the repo
- **Database errors**: Verify `DATABASE_URL` is present and `USE_SQLITE=false`
- **CSV seeding**: Make sure `cibil_database.csv` exists with columns: `CIBIL ID`, `CIBIL Score`

### 11) Security notes
- Never commit secrets or `.env` files
- Use Render Secrets for sensitive values
- Database credentials are automatically managed by Render

### 12) Rollback
- In Render, select a previous successful deploy and roll back

### 13) Fixed Issues
- ✅ Python version compatibility (3.11.4)
- ✅ Pydantic v2 compatibility
- ✅ Package version compatibility
- ✅ Database retry logic
- ✅ Proper error handling
- ✅ All required files included in repo
