import sqlite3

# Connect to the database
conn = sqlite3.connect('nhis.db')
c = conn.cursor()

# List of tables to delete data from
tables = ['hospitals', 'datasets', 'predictions']

# Delete all data from each table
try:
    for table in tables:
        c.execute(f"DELETE FROM {table}")
    conn.commit()
    print("All data has been deleted from the database.")
except sqlite3.Error as e:
    print(f"Error occurred: {str(e)}")
finally:
    # Close the database connection
    conn.close()
