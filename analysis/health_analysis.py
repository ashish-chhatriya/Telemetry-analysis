from analysis.data_loader import load_dataset
import pandas as pd

df = load_dataset()
print("Dataset loaded successfully.")


# Node-level health metrics

node_health = (
    df.groupby("node_id")
      .agg({
          "query_latency": "mean",
          "cpu_utilization": "mean",
          "memory_utilization": "mean",
          "system_load": "mean",
          "active_queries": "mean"
      })
)

print("\nNode Health Metrics:")
print(node_health)


# High-latency threshold

latency_threshold = df["query_latency"].quantile(0.90)


# High-latency rate for each node

high_latency_rate = (
    df.groupby("node_id")["query_latency"]
      .apply(lambda x: (x >= latency_threshold).mean() * 100)
)

node_health["high_latency_rate"] = high_latency_rate


# Normalize metrics to a 0-100 relative risk scale

def normalize(series):
    return (
        (series - series.min()) /
        (series.max() - series.min())
    ) * 100


node_health["latency_score"] = normalize(
    node_health["query_latency"]
)

node_health["cpu_score"] = normalize(
    node_health["cpu_utilization"]
)

node_health["memory_score"] = normalize(
    node_health["memory_utilization"]
)

node_health["load_score"] = normalize(
    node_health["system_load"]
)

node_health["query_load_score"] = normalize(
    node_health["active_queries"]
)

node_health["high_latency_score"] = normalize(
    node_health["high_latency_rate"]
)


# Overall relative risk score

node_health["risk_score"] = (
    node_health["latency_score"] * 0.25 +
    node_health["cpu_score"] * 0.15 +
    node_health["memory_score"] * 0.15 +
    node_health["load_score"] * 0.20 +
    node_health["query_load_score"] * 0.10 +
    node_health["high_latency_score"] * 0.15
)


# Classify relative node risk

def classify_node(score):
    if score >= 75:
        return "Critical"
    elif score >= 50:
        return "Degraded"
    elif score >= 25:
        return "Normal"
    else:
        return "Healthy"


node_health["risk_level"] = node_health["risk_score"].apply(
    classify_node
)


# Final assessment

print("\nNode Risk Assessment:")

print(
    node_health[
        [
            "query_latency",
            "cpu_utilization",
            "memory_utilization",
            "system_load",
            "active_queries",
            "high_latency_rate",
            "risk_score",
            "risk_level"
        ]
    ].sort_values(
        "risk_score",
        ascending=False
    )
)