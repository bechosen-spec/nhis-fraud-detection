import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import bcrypt
import base64
import binascii
import os
import plotly.express as px
from datetime import datetime
from xgboost import XGBClassifier

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="NHIS Fraud Detection",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("assets", exist_ok=True)  # If you have image assets

# --------------------------
# Database Setup
# --------------------------
conn = sqlite3.connect('nhis.db', check_same_thread=False)
c = conn.cursor()

# If 'is_admin' column doesn't exist in 'hospitals', add it
try:
    c.execute("PRAGMA table_info(hospitals)")
    columns = [col[1] for col in c.fetchall()]
    if "is_admin" not in columns:
        c.execute("ALTER TABLE hospitals ADD COLUMN is_admin BOOLEAN DEFAULT 0")
        conn.commit()
except sqlite3.Error as e:
    st.error(f"Database error (adding is_admin): {str(e)}")

# Create tables if not existing
c.execute('''CREATE TABLE IF NOT EXISTS hospitals
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT UNIQUE,
              password TEXT,
              registered_date DATE,
              is_admin BOOLEAN DEFAULT 0)''')

c.execute('''CREATE TABLE IF NOT EXISTS datasets
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              hospital_id INTEGER,
              upload_date DATE,
              data_path TEXT,
              predictions_path TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              dataset_id INTEGER,
              total_cases INTEGER,
              fraud_count INTEGER,
              non_fraud_count INTEGER)''')

# Create default admin user if none exists
try:
    admin_exists = c.execute("SELECT * FROM hospitals WHERE name='admin'").fetchone()
    if not admin_exists:
        hashed_pw_bytes = bcrypt.hashpw('admin123'.encode("utf-8"), bcrypt.gensalt())
        hashed_pw_str = base64.b64encode(hashed_pw_bytes).decode("utf-8")
        c.execute('''
            INSERT INTO hospitals (name, password, registered_date, is_admin)
            VALUES (?,?,?,?)
        ''', ('admin', hashed_pw_str, datetime.now(), True))
        conn.commit()
except sqlite3.Error as e:
    st.error(f"Database error (creating admin): {str(e)}")

# --------------------------
# Model Loading
# --------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/fraud_detection_model.pkl')
    # Do not set `use_label_encoder` here as it's not needed anymore
    scaler = joblib.load('models/scaler.pkl')
    diagnosis_encoder = joblib.load('models/diagnosis_encoder.pkl')
    fraud_encoder = joblib.load('models/fraud_encoder.pkl')
    return model, scaler, diagnosis_encoder, fraud_encoder


model, scaler, diagnosis_encoder, fraud_encoder = load_artifacts()
# --------------------------
# Helper Functions
# --------------------------
def to_int_if_bytes(value):
    """Convert any bytes (possibly from older DB entries) to int, or default to 0 if None."""
    if isinstance(value, bytes):
        return int.from_bytes(value, byteorder='little')
    if value is None:
        return 0
    return int(value)

def preprocess_data(df):
    """
    - Converts 'Date Admitted' and 'Date Discharged' to datetime
    - Computes 'Length of Stay'
    - Encodes Gender, Diagnosis
    - Scales numeric columns
    """
    df = df.copy()
    df['Date Admitted'] = pd.to_datetime(df['Date Admitted'], errors='coerce')
    df['Date Discharged'] = pd.to_datetime(df['Date Discharged'], errors='coerce')

    # Check for invalid or missing date formats
    if df['Date Admitted'].isnull().any() or df['Date Discharged'].isnull().any():
        raise ValueError("Invalid date format in 'Date Admitted' or 'Date Discharged' columns.")

    # Calculate length of stay
    df['Length of Stay'] = (df['Date Discharged'] - df['Date Admitted']).dt.days

    # Remove non-feature columns
    df.drop(['Patient ID', 'Date Admitted', 'Date Discharged'], axis=1, inplace=True, errors='ignore')

    # Encode gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Encode diagnosis
    df['Diagnosis'] = diagnosis_encoder.transform(df['Diagnosis'])

    # Scale numeric columns
    num_cols = ['Age', 'Amount Billed', 'Length of Stay']
    df[num_cols] = scaler.transform(df[num_cols])

    return df

def make_predictions(df):
    processed_df = preprocess_data(df)
    y_pred = model.predict(processed_df)
    return fraud_encoder.inverse_transform(y_pred)

def login_user(name, password):
    """
    1) Query the DB for the user.
    2) If found, decode the DB's base64-encoded password => raw hashed bytes.
    3) Compare with bcrypt.checkpw.
    4) Handle binascii.Error if stored password is invalid base64.
    """
    try:
        user = c.execute("SELECT * FROM hospitals WHERE name=?", (name,)).fetchone()
        if user:
            stored_pw = user[2]  # Could be str or bytes
            if isinstance(stored_pw, bytes):
                stored_pw = stored_pw.decode("utf-8")  # Convert to string

            try:
                stored_pw_bytes = base64.b64decode(stored_pw)
            except binascii.Error:
                st.error("Error: Password hash is not valid base64 (Incorrect padding).")
                return None

            if bcrypt.checkpw(password.encode("utf-8"), stored_pw_bytes):
                return user
        return None
    except sqlite3.Error as e:
        st.error(f"Database error (login): {str(e)}")
        return None

def register_user(name, password):
    """
    1) Bcrypt-hash the user's password => bytes.
    2) Base64-encode => str.
    3) Insert into DB as TEXT.
    """
    try:
        # Basic checks
        if not name or not password:
            st.warning("Please provide both a hospital name and a password.")
            return False

        hashed_pw_bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        hashed_pw_str = base64.b64encode(hashed_pw_bytes).decode("utf-8")
        c.execute('''
            INSERT INTO hospitals (name, password, registered_date)
            VALUES (?,?,?)
        ''', (name, hashed_pw_str, datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("Hospital name already exists. Please choose a different name.")
    except sqlite3.Error as e:
        st.error(f"Database error (register): {str(e)}")
    return False

def local_css(file_name):
    """Load local style.css file for custom CSS if desired."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# Attempt to load CSS
local_css("style.css")

# --------------------------
# Main App
# --------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.logged_in:
        auth_page()
    else:
        if st.session_state.is_admin:
            admin_dashboard()
        else:
            hospital_dashboard()

# --------------------------
# Authentication Page
# --------------------------
def auth_page():
    col1, col2 = st.columns([2, 3])
    
    # Adjust image path to your actual file
    image_path = "assets/image.jpg"
    
    with col1:
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning("⚠️ Image not found. Please check the file path.")

    with col2:
        st.title("NHIS Fraud Detection System")

        auth_type = st.radio("Select Action", ["Login", "Register"], horizontal=True)
        name = st.text_input("Hospital Name")
        password = st.text_input("Password", type="password")

        if auth_type == "Login":
            if st.button("Login 🚀"):
                user = login_user(name, password)
                if user:
                    # user row => [id, name, password, registered_date, is_admin]
                    st.session_state.logged_in = True
                    st.session_state.user_id = user[0]
                    st.session_state.user_name = user[1]
                    st.session_state.is_admin = bool(user[4])
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials or database error.")
        else:
            if st.button("Register ✨"):
                if register_user(name, password):
                    st.success("Registration successful! Please login now.")
                else:
                    st.error("Registration failed. Check logs or messages above.")

# --------------------------
# Hospital Dashboard
# --------------------------
def hospital_dashboard():
    st.title(f"Welcome, {st.session_state.user_name}!")

    # Sidebar logout
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.success("Logged out. Please refresh or re-run to return to login.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📊 Results & Analytics", "ℹ️ Account Info"])

    # 1) Upload Data
    with tab1:
        st.subheader("Upload Patient Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            try:
                df_original = pd.read_csv(uploaded_file)
                required_columns = [
                    "Patient ID", "Date Admitted", "Date Discharged",
                    "Age", "Gender", "Diagnosis", "Amount Billed"
                ]
                missing = [col for col in required_columns if col not in df_original.columns]

                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.success("File uploaded successfully!")

                    if st.button("Run Fraud Detection 🔍"):
                        with st.spinner("Analyzing data..."):
                            try:
                                # Make predictions
                                predictions = make_predictions(df_original)
                                df_predicted = df_original.copy()
                                df_predicted["Prediction"] = predictions

                                # Count how many fraud
                                fraud_count = (df_predicted["Prediction"] != "No Fraud").sum()
                                total_count = len(df_predicted)

                                # Save both the original data & predicted results
                                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                data_path = f"data/{st.session_state.user_id}_{timestamp}.csv"
                                pred_path = f"results/{st.session_state.user_id}_{timestamp}.csv"

                                df_original.to_csv(data_path, index=False)
                                df_predicted.to_csv(pred_path, index=False)

                                # Insert into 'datasets'
                                c.execute("""
                                    INSERT INTO datasets 
                                    (hospital_id, upload_date, data_path, predictions_path)
                                    VALUES (?,?,?,?)
                                """, (st.session_state.user_id, datetime.now(), data_path, pred_path))
                                dataset_id = c.lastrowid

                                # Insert summary in 'predictions'
                                c.execute("""
                                    INSERT INTO predictions
                                    (dataset_id, total_cases, fraud_count, non_fraud_count)
                                    VALUES (?,?,?,?)
                                """, (dataset_id, total_count, fraud_count, total_count - fraud_count))
                                conn.commit()

                                st.balloons()
                                st.success("Analysis completed!")
                                st.write(f"**Total cases:** {total_count}")
                                st.write(f"**Fraud cases:** {fraud_count}")
                                if st.checkbox("Show predicted results"):
                                    st.dataframe(df_predicted)
                            except Exception as e:
                                st.error(f"Error during processing: {str(e)}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    # 2) Results & Analytics
    with tab2:
        st.subheader("Historical Analysis")
        try:
            # Fetch all datasets for the current hospital
            datasets = c.execute("""
                SELECT id, hospital_id, upload_date, data_path, predictions_path
                FROM datasets
                WHERE hospital_id=?
            """, (st.session_state.user_id,)).fetchall()

            if datasets:
                selected_dataset = st.selectbox(
                    "Select Dataset",
                    datasets,
                    format_func=lambda x: f"Dataset #{x[0]} (Uploaded: {x[2]})"
                )
                
                if selected_dataset:
                    # Retrieve predictions summary for this dataset
                    pred_data = c.execute("""
                        SELECT * FROM predictions
                        WHERE dataset_id=?
                    """, (selected_dataset[0],)).fetchone()
                    
                    if pred_data:
                        total_cases_val = to_int_if_bytes(pred_data[2])
                        fraud_count_val = to_int_if_bytes(pred_data[3])
                        non_fraud_count_val = to_int_if_bytes(pred_data[4])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Cases", total_cases_val)
                            st.metric("Potential Fraud Cases", fraud_count_val)

                        with col2:
                            # Pie chart for Fraud vs. Non-Fraud
                            fig = px.pie(
                                names=["Fraud", "Non-Fraud"],
                                values=[fraud_count_val, non_fraud_count_val],
                                title="Fraud Distribution"
                            )
                            st.plotly_chart(fig)

                        # Detailed data?
                        if st.checkbox("Show detailed data"):
                            if selected_dataset[4] and os.path.exists(selected_dataset[4]):
                                df_detail = pd.read_csv(selected_dataset[4])
                                st.dataframe(df_detail)

                                # Breakdown of each fraud label
                                if "Prediction" in df_detail.columns:
                                    st.markdown("### Fraud Breakdown by Type")
                                    fraud_counts = df_detail["Prediction"].value_counts()
                                    breakdown_df = pd.DataFrame({
                                        "Fraud Type": fraud_counts.index,
                                        "Count": fraud_counts.values
                                    })
                                    st.table(breakdown_df)

                                    fig_bar = px.bar(
                                        breakdown_df,
                                        x="Fraud Type",
                                        y="Count",
                                        color="Fraud Type",
                                        title="Fraud Breakdown by Type"
                                    )
                                    st.plotly_chart(fig_bar)
                            else:
                                st.info("Predictions file not found.")
                    else:
                        st.info("No prediction data found for this dataset.")
            else:
                st.info("No datasets uploaded yet.")
        except sqlite3.Error as e:
            st.error(f"Database error (historical analysis): {str(e)}")

    # 3) Account Info
    with tab3:
        st.subheader("Account Information")
        user_data = c.execute("""
            SELECT name, registered_date 
            FROM hospitals
            WHERE id=?
        """, (st.session_state.user_id,)).fetchone()

        if user_data:
            st.write(f"**Hospital Name:** {user_data[0]}")
            st.write(f"**Registration Date:** {user_data[1]}")

        # Change Password
        st.markdown("---")
        st.markdown("## Change Password")
        
        with st.form("change_password_form"):
            old_password = st.text_input("Old Password", type="password", key="old_password")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_password")
            submitted = st.form_submit_button("Change Password")

            if submitted:
                if new_password != confirm_password:
                    st.error("New passwords do not match.")
                else:
                    # Verify old password is correct
                    user = login_user(st.session_state.user_name, old_password)
                    if user:
                        try:
                            hashed_pw_bytes = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
                            hashed_pw_str = base64.b64encode(hashed_pw_bytes).decode("utf-8")

                            c.execute("""
                                UPDATE hospitals
                                SET password=?
                                WHERE id=?
                            """, (hashed_pw_str, st.session_state.user_id))
                            conn.commit()
                            st.success("Password updated successfully!")
                        except sqlite3.Error as e:
                            st.error(f"Database error while updating password: {str(e)}")
                    else:
                        st.error("Old password is incorrect.")

# --------------------------
# Admin Dashboard
# --------------------------
def admin_dashboard():
    st.title("NHIS Admin Dashboard")

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.success("Logged out. Please refresh or re-run to return to login.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🏥 Hospitals", "📈 National Analytics", "🔍 Detailed Fraud Cases"])

    # 1) Registered Hospitals
    with tab1:
        st.subheader("Registered Hospitals")
        try:
            hospitals = c.execute("SELECT * FROM hospitals").fetchall()
            for h in hospitals:
                # h => (id, name, password, registered_date, is_admin)
                with st.expander(f"{h[1]} (Registered: {h[3]})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Hospital ID:** {h[0]}")
                        st.write(f"**Registration Date:** {h[3]}")
                        st.write(f"**Admin Status:** {'Yes' if h[4] else 'No'}")
                    with col2:
                        ds_count = c.execute("""
                            SELECT COUNT(*)
                            FROM datasets
                            WHERE hospital_id=?
                        """, (h[0],)).fetchone()[0]
                        st.write(f"**Datasets Uploaded:** {ds_count}")
        except sqlite3.Error as e:
            st.error(f"Database error (hospitals): {str(e)}")

    # 2) National Analytics
    with tab2:
        st.subheader("National Fraud Overview")
        try:
            # Retrieve all prediction file paths
            c.execute("SELECT predictions_path FROM datasets")
            all_paths = c.fetchall()

            all_frames = []
            for row in all_paths:
                path = row[0]
                if path and os.path.exists(path):
                    df_tmp = pd.read_csv(path)
                    all_frames.append(df_tmp)

            if all_frames:
                combined_df = pd.concat(all_frames, ignore_index=True)
                if "Prediction" in combined_df.columns:
                    total_cases = len(combined_df)
                    total_fraud = combined_df[combined_df["Prediction"] != "No Fraud"].shape[0]

                    if total_cases > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total National Cases", total_cases)
                        with col2:
                            st.metric("Total Fraud Cases", total_fraud)

                        fig = px.pie(
                            names=["Fraud", "Non-Fraud"],
                            values=[total_fraud, total_cases - total_fraud],
                            title="National Fraud Distribution"
                        )
                        st.plotly_chart(fig)

                        fraud_counts = combined_df["Prediction"].value_counts()
                        breakdown_df = pd.DataFrame({
                            "Fraud Type": fraud_counts.index,
                            "Count": fraud_counts.values
                        })
                        st.markdown("### Detailed Fraud Breakdown (National)")
                        st.table(breakdown_df)

                        fig_bar = px.bar(
                            breakdown_df,
                            x="Fraud Type",
                            y="Count",
                            color="Fraud Type",
                            title="Fraud Breakdown by Type (National)"
                        )
                        st.plotly_chart(fig_bar)
                    else:
                        st.info("No data available for analysis.")
                else:
                    st.info("No 'Prediction' column found in combined data.")
            else:
                st.info("No data files found for national fraud analysis.")
        except sqlite3.Error as e:
            st.error(f"Database error (national analytics): {str(e)}")
        except Exception as ex:
            st.error(f"Error loading national data: {str(ex)}")

    # 3) Detailed Fraud Cases
    with tab3:
        st.subheader("Detailed Fraud Cases Across All Hospitals")
        try:
            all_data = []
            all_hosp_ids = c.execute("SELECT id FROM hospitals").fetchall()

            for (hosp_id,) in all_hosp_ids:
                ds_paths = c.execute("""
                    SELECT predictions_path
                    FROM datasets 
                    WHERE hospital_id=?
                """, (hosp_id,)).fetchall()
                for ds in ds_paths:
                    predictions_path = ds[0]
                    if predictions_path and os.path.exists(predictions_path):
                        df_tmp = pd.read_csv(predictions_path)
                        # Filter only fraud cases
                        fraud_rows = df_tmp[df_tmp["Prediction"] != "No Fraud"]
                        if not fraud_rows.empty:
                            all_data.append(fraud_rows)

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                st.markdown("### All Fraud Cases (Combined)")
                st.dataframe(combined_df)
            else:
                st.info("No fraud cases detected across all datasets.")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# --------------------------
# Run the App
# --------------------------
if __name__ == "__main__":
    main()
