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
# High-latency query detection

latency_threshold = df["query_latency"].quantile(0.90)

high_latency = df[df["query_latency"] >= latency_threshold]

print("\nHigh-latency threshold:")
print(latency_threshold)

print("\nNumber of high-latency queries:")
print(len(high_latency))

print("\nHigh-latency queries by node:")
print(high_latency["node_id"].value_counts())

# High-latency rate for each node

total_queries = df.groupby("node_id").size()

high_latency_queries = high_latency.groupby("node_id").size()

high_latency_rate = (
    high_latency_queries
    .div(total_queries)
    .fillna(0)
    .mul(100)
    .sort_values(ascending=False)
)

print("\nHigh-latency rate by node (%):")
print(high_latency_rate)

# Compare normal and high-latency queries

normal_latency = df[df["query_latency"] < latency_threshold]

comparison_metrics = [
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

normal_means = normal_latency[comparison_metrics].mean()
high_latency_means = high_latency[comparison_metrics].mean()

comparison = pd.DataFrame({
    "Normal": normal_means,
    "High_Latency": high_latency_means
})

comparison["Difference"] = (
    comparison["High_Latency"] - comparison["Normal"]
)

comparison["Percentage_Change"] = (
    comparison["Difference"] / comparison["Normal"] * 100
)

print("\nNormal vs High-Latency Conditions:")
print(comparison.sort_values("Percentage_Change", ascending=False))

# Compare high-latency rates and conditions by node

node_latency_comparison = (
    df.groupby(["node_id"])
      .agg(
          avg_latency=("query_latency", "mean"),
          avg_cpu=("cpu_utilization", "mean"),
          avg_memory=("memory_utilization", "mean"),
          avg_system_load=("system_load", "mean"),
          avg_active_queries=("active_queries", "mean"),
          avg_network_latency=("network_latency", "mean"),
          avg_disk_io=("disk_io", "mean")
    )
)

node_latency_comparison["high_latency_rate"] = (
    df.groupby("node_id")["query_latency"]
      .apply(lambda x: (x >= latency_threshold).mean() * 100)
)

print("\nNode Performance and High-Latency Comparison:")
print(node_latency_comparison.sort_values("high_latency_rate", ascending=False))

# Within-node correlation analysis

within_node_correlations = {}

for node in sorted(df["node_id"].unique()):
    node_data = df[df["node_id"] == node]

    correlations = (
        node_data[
            [
                "query_latency",
                "cpu_utilization",
                "memory_utilization",
                "system_load",
                "active_queries",
                "network_latency",
                "disk_io",
                "temperature",
                "error_rate",
                "network_bandwidth"
            ]
        ]
        .corr()["query_latency"]
        .drop("query_latency")
        .sort_values(ascending=False)
    )

    within_node_correlations[node] = correlations

    print(f"\nNode {node} - Correlation with Query Latency:")
    print(correlations)