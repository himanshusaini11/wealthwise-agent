import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
# ------------------


def generate_transactions():
    print("Generating realistic financial data...")

    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=90)

    rows = []

    # Fixed monthly items
    current = start_date
    while current <= end_date:
        if current.day == 1:
            rows.append({"Date": current.strftime("%Y-%m-%d"),
                        "Category": "Rent",
                        "Amount": -1500.00,
                        "Description": "Monthly rent"})
            rows.append({"Date": current.strftime("%Y-%m-%d"),
                        "Category": "Subscriptions",
                        "Amount": -14.99,
                        "Description": "Netflix"})
        current += timedelta(days=1)

    # Daily discretionary spending with trend + weekend effect
    categories = {
        "Food":          (-10, -30),
        "Coffee":        (-4,  -8),
        "Transport":     (-5,  -20),
        "Entertainment": (-20, -80),
        "Shopping":      (-30, -150),
    }

    for day_idx in range(91):
        date = start_date + timedelta(days=day_idx)
        date_str = date.strftime("%Y-%m-%d")
        dow = date.weekday()

        # Trend: base $60/day growing to $105/day over 90 days
        base_spend = 60 + (day_idx * 0.5)

        # Weekend multiplier
        if dow >= 5:
            base_spend *= 1.4

        # Pick 2-4 categories randomly
        num_categories = random.randint(2, 4)
        chosen = random.sample(list(categories.keys()), num_categories)

        # Generate raw amounts
        raw = {}
        for cat in chosen:
            lo, hi = categories[cat]
            raw[cat] = random.uniform(lo, hi)  # negative values

        # Scale to hit base_spend target
        raw_total = sum(abs(v) for v in raw.values())
        scale = base_spend / raw_total if raw_total > 0 else 1.0

        for cat, amount in raw.items():
            scaled_amount = round(amount * scale, 2)
            rows.append({
                "Date": date_str,
                "Category": cat,
                "Amount": scaled_amount,
                "Description": f"{cat} expense",
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("Date").reset_index(drop=True)

    output_path = os.path.join(ROOT_DIR, "data", "transactions.csv")
    df.to_csv(output_path, index=False)

    rent_count = len(df[df["Category"] == "Rent"])
    print(f"Realistic Data saved to: {output_path}")
    print(f"   - Rent entries: {rent_count}")
    print(f"   - Total rows: {len(df)}")
    print(f"   - Date range: {df['Date'].min()} to {df['Date'].max()}")


if __name__ == "__main__":
    generate_transactions()
