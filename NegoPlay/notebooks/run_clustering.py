"""Quick script to run clustering and print results."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from src.shared.data_loader import load_matches
from src.stage1_clustering.features import compute_player_features
from src.stage1_clustering.clustering import cluster_players, FEATURE_COLS

DATA = (
    r"C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה"
    r"\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
    r"\data\processed\all_matches_full.csv"
)

print("Loading data...")
df = load_matches(DATA)
features = compute_player_features(df, min_bidding_boards=20)
print(f"Players: {len(features)}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print()

result, info = cluster_players(features)

print()
print("=== RESULTS ===")
pv = [round(v*100, 1) for v in info["pca_explained_variance"]]
print(f"PCA variance per component : {pv}%")
print(f"PCA cumulative (3 PCs)     : {round(info['pca_cumulative_variance']*100, 1)}%")
print()
print("Silhouette scores per k (PCA space, 8 features):")
for k, s in sorted(info["silhouette_scores"].items()):
    flag = " <-- BEST" if k == info["best_k"] else ""
    print(f"  k={k}  silhouette={s:.4f}{flag}")
print()
print(f"Best silhouette:  {info['kmeans_silhouette']:.4f}")
print(f"HDBSCAN clusters: {info['hdbscan_n_clusters']}")
print(f"Cluster sizes:    {info['cluster_sizes']}")

# Compare to old result (10 features, no PCA)
print()
print("=== COMPARISON ===")
print("Old (10 features, no PCA):  silhouette ~ 0.15")
print(f"New (8 features + PCA):     silhouette = {info['kmeans_silhouette']:.4f}")
