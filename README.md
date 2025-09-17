# NHIS Billing Fraud Detection (Streamlit App)

A Streamlit-powered web application that helps hospitals and the NHIS team detect potential billing fraud from inpatient claims. The app supports secure hospital registration & login, CSV uploads, model-backed predictions, analytics dashboards, and an admin view for national roll-ups.

**Live demo:** `https://nhis-fraud-detection.streamlit.app`
*(If the demo is down, run locally using the steps below.)*

---

## Table of Contents

* [Features](#features)
* [Architecture at a Glance](#architecture-at-a-glance)
* [Folder Structure](#folder-structure)
* [Requirements](#requirements)
* [Quick Start (Local)](#quick-start-local)
* [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
* [Data Format](#data-format)
* [Model Artifacts](#model-artifacts)
* [How Inference Works](#how-inference-works)
* [Admin Account](#admin-account)
* [Security Notes](#security-notes)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

---

## Features

* **Authentication**

  * Hospital registration & login
  * Passwords hashed with `bcrypt` and stored base64-encoded in SQLite
  * Admin role for national analytics

* **Data Upload & Inference**

  * Upload inpatient claims as CSV
  * Validates required columns & dates
  * Computes “Length of Stay”
  * Normalizes categories (e.g., Gender), scales numeric features
  * Predicts fraud label per case using a trained ML model
  * One-click **Download predictions** as CSV

* **Analytics**

  * Per-hospital dataset history and summary
  * Fraud vs Non-Fraud distribution (pie & bar charts)
  * National view (admin) aggregating all hospitals
  * Detailed listing of predicted fraud cases

* **Resilience**

  * Safer model artifact loading with clear errors
  * Optional `feature_names.joblib` for stable inference
  * Graceful handling for unseen categories (e.g., Diagnosis)

---

## Architecture at a Glance

* **Frontend/UI:** Streamlit (`app.py`)
* **Storage:** SQLite (`nhis.db`)
* **ML Runtime:** `scikit-learn`/`xgboost` (model loaded from `/models`)
* **Artifacts:** Scaler, encoders, trained model, feature column order
* **Visuals:** Plotly charts
* **Assets:** Local images & optional sample CSV

---

## Folder Structure

```
.
├─ .devcontainer/
├─ assets/
│  ├─ image.jpg
│  └─ image.png
├─ models/
│  ├─ best_xgb_model.pkl          # or fraud_detection_model.pkl
│  ├─ diagnosis_encoder.pkl
│  ├─ fraud_encoder.pkl
│  ├─ scaler.pkl
│  └─ feature_names.joblib        # recommended (see below)
├─ app.py
├─ delete_all_data.py
├─ new.py
├─ nhis.db
├─ requirements.txt
├─ style.css
└─ README.md
```

> Tip: Keep **one** canonical model (e.g., `best_xgb_model.pkl`) to avoid confusion.

---

## Requirements

* Python 3.10+
* See `requirements.txt`:

  ```
  streamlit>=1.36
  pandas>=2.0
  numpy>=1.24
  scikit-learn>=1.3
  joblib>=1.3
  bcrypt>=4.1
  plotly>=5.22
  xgboost>=2.0
  ```

---

## Quick Start (Local)

1. **Clone & install**

   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   python -m pip install -r requirements.txt
   ```

2. **Ensure model artifacts exist** under `models/` (see [Model Artifacts](#model-artifacts)).

3. *(Optional but recommended)* Move SQLite into `data/` for tidiness:

   * Create folder `data/`
   * Update `app.py` connection to:

     ```python
     os.makedirs("data", exist_ok=True)
     conn = sqlite3.connect('data/nhis.db', check_same_thread=False)
     ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Open the URL printed in the terminal (usually `http://localhost:8501`).

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), select your repo & branch.
3. Set **Main file path** to `app.py`.
4. Ensure `requirements.txt` is present.
5. Deploy.
   *Note:* Streamlit Cloud’s filesystem is ephemeral; `nhis.db` can reset on redeploy. For production, use a managed DB (e.g., Postgres/Supabase/Neon).

---

## Data Format

**Required CSV columns:**

* `Patient ID` *(string)*
* `Date Admitted` *(YYYY-MM-DD preferred)*
* `Date Discharged` *(YYYY-MM-DD preferred)*
* `Age` *(integer/float)*
* `Gender` *("Male"/"Female"; variants like "M"/"F" are normalized)*
* `Diagnosis` *(string)*
* `Amount Billed` *(numeric)*

**Example:**

```csv
Patient ID,Date Admitted,Date Discharged,Age,Gender,Diagnosis,Amount Billed
P001,2025-08-01,2025-08-05,45,Male,Malaria,50000
P002,2025-08-03,2025-08-04,30,Female,Typhoid,35000
```

**Notes**

* The app computes **Length of Stay** = `Date Discharged - Date Admitted` (days).
* Negative stays or invalid dates will raise a friendly error.

---

## Model Artifacts

Place these in the `models/` directory:

* `best_xgb_model.pkl` **or** `fraud_detection_model.pkl` (trained model)
* `scaler.pkl` (fitted `StandardScaler` or equivalent)
* `diagnosis_encoder.pkl` (LabelEncoder/OrdinalEncoder for `Diagnosis`)
* `fraud_encoder.pkl` (LabelEncoder for the target)
* `feature_names.joblib` (**recommended**) – list of **final** training columns in exact order
  *(Prevents inference-time column/order mismatches)*

**Export `feature_names.joblib` during training** (after all preprocessing and just before `model.fit`):

```python
import joblib
feature_names = list(X.columns)  # X is your final training matrix
joblib.dump(feature_names, 'models/feature_names.joblib')
```

*(Optional)* `label_map.joblib` if you want to rename raw model labels to nicer UI labels.
The app currently maps `"Fake Treatment" → "Wrong Diagnosis"` by default.

---

## How Inference Works

1. **Upload CSV** → validate required columns.
2. **Date parsing & Length of Stay** computation.
3. **Categorical handling**

   * `Gender` normalized to {0,1}
   * `Diagnosis` encoded via `diagnosis_encoder`
     *(Unseen values are handled gracefully where possible.)*
4. **Scaling** of numerics (e.g., `Age`, `Amount Billed`, `Length of Stay`) via `scaler.pkl`.
5. **Column alignment** using `feature_names.joblib` (if available).
   Missing columns → filled with 0; extra columns are dropped.
6. **Model prediction** → inverse transform with `fraud_encoder`.
7. **UI label mapping** (e.g., rename `"Fake Treatment"` to `"Wrong Diagnosis"`).
8. **Save & Download** results and log dataset metadata to SQLite.

---

## Admin Account

On first run, the app creates a default admin user:

* **Username:** `admin`
* **Password:** `admin123`

> 🔒 **Important:** Log in and change this password immediately via the **Account Info → Change Password** form.

---

## Security Notes

* Passwords are hashed with `bcrypt` (random salt) and stored base64-encoded in SQLite.
* SQLite has no true boolean type; `is_admin` is stored as integer `0/1` and converted to Python `bool`.
* For production:

  * Use managed DB with TLS
  * Rotate admin password
  * Consider JWT sessions or OAuth
  * Restrict file types and add server-side CSV sanitation if exposed publicly

---

## Troubleshooting

**Common issues & fixes:**

* **App can’t find model artifacts**

  * Ensure files exist under `/models` and names match exactly.
  * Keep only **one** primary model file (e.g., `best_xgb_model.pkl`).

* **Column mismatch / `ValueError` on transform**

  * Export `feature_names.joblib` from training and place in `/models`.
  * Ensure your uploaded CSV columns match the documented schema.

* **Unseen `Diagnosis` values**

  * Add an `"Other"` category during training or upgrade unseen handling logic.

* **Invalid dates or negative Length of Stay**

  * Fix dates in CSV; the app will reject invalid records.

* **Streamlit Community Cloud resets data**

  * Use an external DB for persistence; SQLite files may reset on redeploy.

* **Charts not rendering**

  * Ensure there is at least one dataset uploaded with predictions.

**Debug panel (optional):**
A small “System Health” expander can be added in `app.py` to list file presence, DB connectivity, and artifact status. (See code in issue discussions or comments within `app.py` if included.)

---

## Roadmap

* Role-based dashboards (regional/state health authorities)
* Model monitoring (data drift, performance over time)
* Multi-label fraud categories & severity scoring
* Audit trail download (CSV/Excel/PDF)
* External DB support (Postgres) with migrations
* Batch API endpoint for programmatic submissions
* Better unseen-category handling via target encoders / embeddings

---

## Contributing

Contributions are welcome!

1. Fork the repo & create a feature branch:

   ```bash
   git checkout -b feat/my-improvement
   ```
2. Make changes and add tests where possible.
3. Submit a PR describing the change and any migration steps.

---

## License

Choose a license and add it to the repository (e.g., MIT, Apache-2.0).
Until then, assume **All Rights Reserved**.

---

## Acknowledgements

* Streamlit for the rapid UI layer
* scikit-learn / XGBoost for modeling
* Plotly for interactive charts

---

### Maintainer Notes

If you change your training pipeline:

* Re-export `diagnosis_encoder.pkl`, `fraud_encoder.pkl`, `scaler.pkl`, the **trained model**, and **`feature_names.joblib`**.
* Keep the inference preprocessing **identical** to training (feature names, order, encodings).
