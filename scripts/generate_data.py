import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
import os

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "transactions.csv")
os.makedirs(DATA_DIR, exist_ok=True)
# ------------------

fake = Faker()

def generate_transactions():
    print("Generating realistic financial data...")
    
    data = []
    # 90-day window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # 1. FIXED MONTHLY BILLS (The "Rent" Fix)
    # Iterate through each month in the window
    current = start_date
    while current <= end_date:
        # If it's the 1st of the month, pay Rent
        if current.day == 1:
            data.append({
                "Date": current.strftime("%Y-%m-%d"),
                "Category": "Rent",
                "Amount": -1500.00,
                "Description": "Monthly Apartment Rent"
            })
            # Also pay a Subscription
            data.append({
                "Date": current.strftime("%Y-%m-%d"),
                "Category": "Subscriptions",
                "Amount": -14.99,
                "Description": "Netflix Premium"
            })
        current += timedelta(days=1)

    # 2. DAILY DISCRETIONARY SPENDING (Randomized)
    # These happen randomly (e.g., 2-4 times a day)
    discretionary_cats = {
        'Food': (-30, -10),
        'Entertainment': (-80, -20),
        'Transport': (-20, -5),
        'Shopping': (-150, -30),
        'Coffee': (-8, -4)
    }
    
    # Loop through every single day
    current = start_date
    while current <= end_date:
        # Randomly decide how many transactions today (0 to 3)
        num_transactions = random.randint(0, 3)
        
        for _ in range(num_transactions):
            category = random.choice(list(discretionary_cats.keys()))
            min_amt, max_amt = discretionary_cats[category]
            amount = round(random.uniform(min_amt, max_amt), 2)
            
            data.append({
                "Date": current.strftime("%Y-%m-%d"),
                "Category": category,
                "Amount": amount,
                "Description": fake.sentence(nb_words=4)
            })
        current += timedelta(days=1)

    # 3. Save Sorted Data
    df = pd.DataFrame(data)
    df = df.sort_values(by="Date")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Realistic Data saved to: {OUTPUT_PATH}")
    print(f"   - Rent entries: {len(df[df['Category']=='Rent'])}")
    print(f"   - Total rows: {len(df)}")

if __name__ == "__main__":
    generate_transactions()