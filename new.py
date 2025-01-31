#!/usr/bin/env python3

import sqlite3
import time
import bcrypt
import base64
from datetime import datetime

DB_PATH = "nhis.db"
BUSY_TIMEOUT_MS = 3000
RETRY_ATTEMPTS = 5
RETRY_DELAY_SEC = 1

def execute_with_retry(conn, cursor, query, params=()):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            cursor.execute(query, params)
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                print(f"Database locked. Retrying {attempt+1}/{RETRY_ATTEMPTS} in {RETRY_DELAY_SEC} second(s)...")
                time.sleep(RETRY_DELAY_SEC)
            else:
                raise
    raise Exception("Database still locked after retries")

def add_new_admin():
    name = input("Enter the Hospital Name: ")
    password = input("Enter the Password: ")

    hashed_pw_bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    hashed_pw_str = base64.b64encode(hashed_pw_bytes).decode("utf-8")

    query = """
        INSERT INTO hospitals (name, password, registered_date, is_admin)
        VALUES (?,?,?,?)
    """
    params = (name, hashed_pw_str, datetime.now(), True)
    
    try:
        execute_with_retry(conn, c, query, params)
        print(f"New admin '{name}' added successfully!")
    except sqlite3.IntegrityError:
        print("Error: A hospital with this name already exists.")
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # Set a busy timeout
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")

    # (Optional) Use WAL mode
    # conn.execute("PRAGMA journal_mode = WAL")

    c = conn.cursor()

    add_new_admin()

    # Close resources
    c.close()
    conn.close()
