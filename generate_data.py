"""Generate synthetic AWS cost and usage data with a few planted anomalies.

Run:  python generate_data.py
Output: aws_costs.csv  (one row per day, per service, per region)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed so results are repeatable

# Typical daily base cost (in USD) for each service
SERVICES = {
    "EC2": 120.0,
    "S3": 35.0,
    "RDS": 80.0,
    "Lambda": 12.0,
    "CloudWatch": 18.0,
    "DataTransfer": 25.0,
}
REGIONS = ["ap-south-1", "us-east-1"]

dates = pd.date_range("2026-08-01", periods=90, freq="D")

rows = []
for date in dates:
    # Weekends are slightly cheaper (less usage)
    weekend_factor = 0.85 if date.dayofweek >= 5 else 1.0
    for service, base in SERVICES.items():
        for region in REGIONS:
            region_factor = 1.0 if region == "ap-south-1" else 0.6
            noise = rng.normal(1.0, 0.05)  # about 5% random variation
            cost = base * region_factor * weekend_factor * noise
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "service": service,
                "region": region,
                "cost_usd": round(cost, 2),
            })

df = pd.DataFrame(rows)

# Planted anomalies: the agent should be able to find these later
df.loc[(df.date == "2026-09-12") & (df.service == "EC2") & (df.region == "ap-south-1"), "cost_usd"] *= 4
df.loc[(df.date == "2026-10-03") & (df.service == "DataTransfer") & (df.region == "us-east-1"), "cost_usd"] *= 6
df.loc[(df.date == "2026-10-20") & (df.service == "Lambda") & (df.region == "ap-south-1"), "cost_usd"] *= 8

df["cost_usd"] = df["cost_usd"].round(2)
df.to_csv("aws_costs.csv", index=False)
print(f"Saved {len(df)} rows to aws_costs.csv")
print(df.groupby("service")["cost_usd"].sum().round(2))
