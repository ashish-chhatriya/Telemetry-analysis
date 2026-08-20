from analysis.data_loader import load_dataset
import pandas as pd

df = load_dataset()

print("Dataset loaded successfully.")


# metrics to average for each node
metrics = [
    "cpu_utilization",
    "memory_utilization",
    "disk_io",
    "network_latency",
    "network_bandwidth",
    "active_queries",
    "system_load",
    "temperature",
    "error_rate"
]


# calculating representative metric for each node 
resource_summary = (
    df.groupby("node_id")[metrics]
      .mean()
)


#latency summary for each node
latency_summary = (
    df.groupby("node_id")["query_latency"]
      .agg(["mean", "median", "max"])
)


print("\nResource/System Summary:")
print(resource_summary)

print("\nQuery Latency Summary:")
print(latency_summary)


# Rank nodes by average query latency
latency_ranking = (
    df.groupby("node_id")["query_latency"]
      .mean()
      .sort_values(ascending=False)
)


print("\nNodes ranked by average query latency:")
print(latency_ranking)

print("\nCorrelation with Query Latency:")

correlations = (
    df.corr(numeric_only=True)["query_latency"]
      .sort_values(ascending=False)
)

print(correlations)