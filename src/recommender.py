"""
Data Mining Assignment 2 – Recommender Systems
Models : User-based KNN  (cornac.UserKNN)
         SVD Matrix Factorisation (cornac.SVD – Simon Funk's SVD)
Library: Cornac  (pip install cornac)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cornac
from cornac.eval_methods import CrossValidation
from cornac.models      import UserKNN, SVD
from cornac.metrics     import RMSE, Precision, Recall
import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────

movies   = pd.read_csv("../data/processed/movies_processed.csv")
train_df = pd.read_csv("../data/processed/ratings_processed.csv")
test_df  = pd.read_csv("../data/raw/ratings_test.csv")

print(f"Train : {train_df.shape[0]:,} ratings | "
      f"{train_df.userId.nunique()} users | "
      f"{train_df.movieId.nunique()} movies")
print(f"Rating range : {train_df.rating.min()} – {train_df.rating.max()}")

# Cornac expects (user, item, rating) as strings or ints
uir_data = list(zip(
    train_df.userId.astype(str),
    train_df.movieId.astype(str),
    train_df.rating.astype(float)
))

train_user_ids  = set(train_df.userId.unique())
all_movie_ids   = sorted(train_df.movieId.unique())

# ─── 2. DEFINE MODELS ─────────────────────────────────────────────────────────

ucf_model = UserKNN(
    k            = 40,
    similarity   = "cosine",
    mean_centered= True,      # equivalent to our mean-centred cosine CF
    name         = "UserKNN",
    verbose      = False,
    seed         = 42,
)

svd_model = SVD(
    k          = 50,          # latent factors
    max_iter   = 20,
    learning_rate = 0.005,
    lambda_reg = 0.02,
    name       = "SVD",
    verbose    = False,
    seed       = 42,
)

# ─── 3. CROSS-VALIDATION WITH CORNAC ──────────────────────────────────────────

print("\n" + "="*60)
print("CROSS-VALIDATION (5-fold)")
print("="*60)

cv = CrossValidation(
    data             = uir_data,
    n_folds          = 5,
    rating_threshold = 3.5,   # threshold for Precision / Recall
    exclude_unknowns = True,
    seed             = 42,
    verbose          = False,
)

exp = cornac.Experiment(
    eval_method = cv,
    models      = [ucf_model, svd_model],
    metrics     = [RMSE(), Precision(k=10), Recall(k=10)],
    user_based  = True,
    verbose     = False,
)

exp.run()

# ─── 4. EXTRACT & DISPLAY RESULTS ─────────────────────────────────────────────

# exp.result is a list of CVResult objects, one per model
# Each CVResult has: .model_name, .metric_mean (dict), .metric_std (dict)
print("\n--- Average Results ---")
result_rows = []
for cv_result in exp.result:
    row = {"Model": cv_result.model_name}
    row.update({k: round(v, 4) for k, v in cv_result.metric_mean.items()})
    result_rows.append(row)
    print(f"{cv_result.model_name:10s} | " +
          " | ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "Model"))

results_df = pd.DataFrame(result_rows).set_index("Model")

# Per-fold scores: each cv_result[fold_i] is a Result with .metric_avg_results
fold_data = {r.model_name: {m: [] for m in r.metric_mean} for r in exp.result}
for cv_result in exp.result:
    for fold_result in cv_result:
        for metric_name, score in fold_result.metric_avg_results.items():
            if metric_name in fold_data[cv_result.model_name]:
                fold_data[cv_result.model_name][metric_name].append(score)

# ─── 5. VISUALISATIONS ────────────────────────────────────────────────────────

TIMING_KEYS   = {"Train (s)", "Test (s)"}
metric_keys   = [k for k in list(fold_data.values())[0].keys() if k not in TIMING_KEYS]
metric_titles = [f"{m} {'(lower is better)' if 'RMSE' in m else ''}" for m in metric_keys]
colors        = ["#4C72B0", "#DD8452"]
model_names   = [r.model_name for r in exp.result]

fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 4))
fig.suptitle("Model Comparison – Cornac 5-fold CV", fontsize=13)

for ax, mkey, mtitle in zip(axes, metric_keys, metric_titles):
    box_data = [fold_data[mn][mkey] for mn in model_names]
    bp = ax.boxplot(box_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(mtitle)
    ax.set_ylabel(mkey)
    ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("../figures/model_comparison.png", dpi=150)
plt.close()
print("\nSaved: figures/model_comparison.png")

# Rating distribution
fig, ax = plt.subplots(figsize=(7, 4))
train_df["rating"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color="#4C72B0", alpha=0.8)
ax.set_title("Rating Distribution in Training Set")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../figures/rating_distribution.png", dpi=150)
plt.close()
print("Saved: figures/rating_distribution.png")


# ─── 6. TRAIN FINAL MODEL ON FULL DATA ────────────────────────────────────────

print("\n" + "="*60)
print("TRAINING FINAL MODEL ON FULL TRAINING SET")
print("="*60)

full_dataset = cornac.data.Dataset.from_uir(uir_data, seed=42)

svd_final = SVD(
    k             = 50,
    max_iter      = 20,
    learning_rate = 0.005,
    lambda_reg    = 0.02,
    name          = "SVD",
    verbose       = False,
    seed          = 42,
)
svd_final.fit(full_dataset)
print("SVD fitted on full training set.")


# ─── 7. TASK 2 – TOP-10 RECOMMENDATIONS ───────────────────────────────────────

print("\n" + "="*60)
print("TASK 2: GENERATING RECOMMENDATIONS")
print("="*60)

# Popularity fallback for cold-start users
movie_popularity = (
    train_df.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .query("count >= 10")
    .sort_values("mean", ascending=False)
)
popular_movie_ids = movie_popularity.index.tolist()

test_user_ids    = sorted(test_df.userId.tolist())
cold_start_users = set(test_user_ids) - train_user_ids
print(f"Cold-start users ({len(cold_start_users)}): {sorted(cold_start_users)}")

recommendations = {}

for uid in test_user_ids:
    if uid in cold_start_users:
        # No training history → popular movies fallback
        recs = popular_movie_ids[:10]
    else:
        uid_str = str(uid)
        # recommend() excludes already-rated items when remove_seen=True
        # recommend() returns a list of string item IDs, already excluding seen items
        top_items = svd_final.recommend(
            user_id     = uid_str,
            k           = 10,
            remove_seen = True,
            train_set   = full_dataset,
        )
        recs = [int(iid) for iid in top_items]

        # Pad with popular movies if needed
        if len(recs) < 10:
            already_rated = set(train_df[train_df.userId == uid].movieId.tolist())
            for pm in popular_movie_ids:
                if pm not in set(recs) and pm not in already_rated:
                    recs.append(pm)
                if len(recs) == 10:
                    break

    recommendations[uid] = recs
    print(f"  User {uid:4d}: {recs}")


# ─── 8. SAVE ratings_test.csv ─────────────────────────────────────────────────

output = test_df.copy()
for uid, recs in recommendations.items():
    row_mask = output.userId == uid
    for i, mid in enumerate(recs[:10], start=1):
        output.loc[row_mask, f"recommendation{i}"] = mid

output.to_csv("../data/raw/ratings_test.csv", index=False)
print("\nSaved: data/raw/ratings_test.csv (filled)")
print("Done.")
