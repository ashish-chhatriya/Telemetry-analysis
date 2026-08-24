from analysis.data_loader import load_dataset
import pandas as pd

df = load_dataset()

# High-latency threshold
latency_threshold = df["query_latency"].quantile(0.90)

# Node-level metrics
dashboard = (
    df.groupby("node_id")
      .agg(
          avg_latency=("query_latency", "mean"),
          avg_cpu=("cpu_utilization", "mean"),
          avg_memory=("memory_utilization", "mean"),
          avg_system_load=("system_load", "mean"),
          avg_active_queries=("active_queries", "mean")
      )
)

# High-latency rate
high_latency_rate = (
    df.groupby("node_id")["query_latency"]
      .apply(lambda x: (x >= latency_threshold).mean() * 100)
)

dashboard["high_latency_rate"] = high_latency_rate

# Normalize
def normalize(series):
    return (
        (series - series.min()) /
        (series.max() - series.min())
    ) * 100

dashboard["latency_score"] = normalize(dashboard["avg_latency"])
dashboard["cpu_score"] = normalize(dashboard["avg_cpu"])
dashboard["memory_score"] = normalize(dashboard["avg_memory"])
dashboard["load_score"] = normalize(dashboard["avg_system_load"])
dashboard["query_load_score"] = normalize(dashboard["avg_active_queries"])
dashboard["high_latency_score"] = normalize(
    dashboard["high_latency_rate"]
)

# Risk score
dashboard["risk_score"] = (
    dashboard["latency_score"] * 0.25 +
    dashboard["cpu_score"] * 0.15 +
    dashboard["memory_score"] * 0.15 +
    dashboard["load_score"] * 0.20 +
    dashboard["query_load_score"] * 0.10 +
    dashboard["high_latency_score"] * 0.15
)

# Risk level
def classify_node(score):
    if score >= 75:
        return "Critical"
    elif score >= 50:
        return "Degraded"
    elif score >= 25:
        return "Normal"
    else:
        return "Healthy"

dashboard["risk_level"] = dashboard["risk_score"].apply(
    classify_node
)

dashboard = dashboard.reset_index()

# Export for Power BI
dashboard.to_csv(
    "dashboard_node_data.csv",
    index=False
)

print("Power BI dataset created:")
print(dashboard)
