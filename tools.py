"""Deterministic cost-analysis tools.

The language model never does arithmetic. It only decides WHICH tool to call
and explains the result. All numbers come from pandas.
"""
import pandas as pd

DF = pd.read_csv("aws_costs.csv", parse_dates=["date"])


def total_cost(service: str = "", start_date: str = "", end_date: str = "") -> dict:
    """Total spend. Optionally filter by one service name (e.g. EC2).

    IMPORTANT: leave start_date and end_date EMPTY unless the user explicitly
    mentions dates. Dates must be YYYY-MM-DD.
    """
    d = DF
    if service:
        d = d[d.service.str.lower() == service.lower()]
    if start_date or end_date:
        lo, hi = DF.date.min(), DF.date.max()
        # Guard: the model sometimes invents dates outside the data. Say so clearly.
        if (start_date and pd.Timestamp(start_date) > hi) or (end_date and pd.Timestamp(end_date) < lo):
            return {"error": f"Requested dates are outside the available data, which covers "
                             f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}. Retry without dates or within this range."}
    if start_date:
        d = d[d.date >= start_date]
    if end_date:
        d = d[d.date <= end_date]
    return {"service": service or "ALL", "start": start_date or "beginning",
            "end": end_date or "end", "total_usd": round(float(d.cost_usd.sum()), 2)}


def top_services(n: int = 3) -> list:
    """The n most expensive services overall."""
    s = DF.groupby("service").cost_usd.sum().sort_values(ascending=False).head(int(n))
    return [{"service": k, "total_usd": round(float(v), 2)} for k, v in s.items()]


def cost_trend(service: str = "") -> dict:
    """Monthly totals and month-over-month % change."""
    d = DF[DF.service.str.lower() == service.lower()] if service else DF
    m = d.groupby(d.date.dt.to_period("M")).cost_usd.sum()
    out = []
    prev = None
    for period, total in m.items():
        change = None if prev is None else round((total - prev) / prev * 100, 1)
        out.append({"month": str(period), "total_usd": round(float(total), 2),
                    "mom_change_pct": change})
        prev = total
    return {"service": service or "ALL", "months": out}


def detect_anomalies(z_threshold: float = 3.0) -> list:
    """Find days where a service+region cost is unusually far from its own average.

    z-score = (value - mean) / standard deviation, computed per service+region.
    """
    g = DF.groupby(["service", "region"]).cost_usd
    z = (DF.cost_usd - g.transform("mean")) / g.transform("std")
    hits = DF[z.abs() > z_threshold].copy()
    hits["z_score"] = z[hits.index].round(1)
    hits["date"] = hits.date.dt.strftime("%Y-%m-%d")
    return hits[["date", "service", "region", "cost_usd", "z_score"]].to_dict("records")


if __name__ == "__main__":
    print(total_cost("EC2"))
    print(top_services(3))
    print(cost_trend("Lambda"))
    print(detect_anomalies())