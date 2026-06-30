"""
generate_data.py
Generates a synthetic, PaySim-style transaction dataset for fraud detection.

Why synthetic data:
- Lets the whole pipeline run end-to-end with zero manual downloads.
- Mimics the structure of real fraud datasets (IEEE-CIS / PaySim), so swapping
  in the real dataset later only requires changing the CSV path + column names
  in train_model.py, not the pipeline logic.

Usage:
    python generate_data.py --n_rows 200000 --out ../data/transactions.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_transactions(n_rows: int, fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    transaction_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_weights = [0.35, 0.20, 0.20, 0.15, 0.10]

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    def make_block(n, is_fraud):
        amount = rng.lognormal(mean=4.0 if not is_fraud else 6.0, sigma=1.2, size=n)
        old_balance_orig = rng.uniform(0, 50000, size=n)
        # Fraudulent transfers more often drain the account close to zero
        if is_fraud:
            new_balance_orig = np.clip(old_balance_orig - amount, 0, None) * rng.uniform(0, 0.1, size=n)
        else:
            new_balance_orig = np.clip(old_balance_orig - amount, 0, None)

        old_balance_dest = rng.uniform(0, 50000, size=n)
        new_balance_dest = old_balance_dest + amount * rng.uniform(0.8, 1.0, size=n)

        ttype = rng.choice(
            ["TRANSFER", "CASH_OUT"] if is_fraud else transaction_types,
            size=n,
            p=[0.6, 0.4] if is_fraud else type_weights,
        )

        step = rng.integers(1, 745, size=n)  # hour-of-month style timestamp, like PaySim
        hour_of_day = step % 24

        return pd.DataFrame({
            "step": step,
            "hour_of_day": hour_of_day,
            "type": ttype,
            "amount": amount,
            "oldbalanceOrg": old_balance_orig,
            "newbalanceOrig": new_balance_orig,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
            "isFraud": int(is_fraud),
        })

    df = pd.concat([make_block(n_legit, False), make_block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    df["transaction_id"] = [f"TXN{1000000 + i}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=200_000)
    parser.add_argument("--fraud_rate", type=float, default=0.012)  # ~1.2%, realistic class imbalance
    parser.add_argument("--out", type=str, default="transactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_transactions(args.n_rows, args.fraud_rate, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['isFraud'].sum():,} fraud, {df['isFraud'].mean()*100:.2f}%) to {args.out}")
