import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# FINAL ANALYTIC QUESTION:
# Do teams with above-median tournament possession have a significantly higher
# mean number of shots on target per 90 minutes than teams with
# below/equal-median possession?

# 1. Load the full dataset
df = pd.read_csv("world_cup_2026_possession_shots_full.csv")

# 2. Data wrangling / checks
print(df.head())
print(df.info())

print("\nMissing values:")
print(df[["Team", "Possession_Percentage", "SoT_per_90"]].isnull().sum())

# 3. Simple random sample
# Population = 48 World Cup teams
# Sample = 36 teams
# A fixed random seed makes the sample reproducible
sample = df.sample(n=36, random_state=140).copy()

# 4. Calculate the median tournament possession for the sample
median_possession = sample["Possession_Percentage"].median()

# Teams above the median are placed in the "Above-median possession" group.
# Teams equal to or below the median are placed in the
# "Below/equal-median possession" group.
sample["Possession_Group"] = np.where(
    sample["Possession_Percentage"] > median_possession,
    "Above-median possession",
    "Below/equal-median possession"
)

print("\nSample median tournament possession:", median_possession)
print("\nGroup sizes:")
print(sample["Possession_Group"].value_counts())

# 5. Descriptive statistics
descriptive = sample.groupby("Possession_Group")["SoT_per_90"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print("\nDescriptive statistics:")
print(descriptive)

above = sample.loc[
    sample["Possession_Group"] == "Above-median possession",
    "SoT_per_90"
]

below_equal = sample.loc[
    sample["Possession_Group"] == "Below/equal-median possession",
    "SoT_per_90"
]

# 6. Welch independent two-sample t-test
# H0: mean SoT/90 for above-median possession teams
#     <= mean SoT/90 for below/equal-median possession teams
#
# H1: mean SoT/90 for above-median possession teams
#     > mean SoT/90 for below/equal-median possession teams

t_stat, p_two_sided = stats.ttest_ind(
    above,
    below_equal,
    equal_var=False
)

# Convert the two-sided p-value to the required one-sided p-value
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

print("\nWelch t-statistic:", t_stat)
print("One-sided p-value:", p_one_sided)

alpha = 0.05

if p_one_sided < alpha:
    print("Decision: Reject H0.")
    print(
        "Conclusion: Above-median possession teams have a significantly "
        "higher mean number of shots on target per 90 minutes than "
        "below/equal-median possession teams."
    )
else:
    print("Decision: Fail to reject H0.")
    print(
        "Conclusion: There is insufficient evidence that above-median "
        "possession teams have a higher mean number of shots on target "
        "per 90 minutes than below/equal-median possession teams."
    )

# 7. 95% confidence interval for the difference in means
# Difference = above-median mean - below/equal-median mean

mean_diff = above.mean() - below_equal.mean()

se = np.sqrt(
    above.var(ddof=1) / len(above)
    + below_equal.var(ddof=1) / len(below_equal)
)

welch_df = (
    (above.var(ddof=1) / len(above)
     + below_equal.var(ddof=1) / len(below_equal)) ** 2
    /
    (
        (above.var(ddof=1) / len(above)) ** 2 / (len(above) - 1)
        + (below_equal.var(ddof=1) / len(below_equal)) ** 2
        / (len(below_equal) - 1)
    )
)

ci_low, ci_high = stats.t.interval(
    0.95,
    welch_df,
    loc=mean_diff,
    scale=se
)

print("\nMean difference (above-median - below/equal-median):", mean_diff)
print("95% confidence interval:", (ci_low, ci_high))

# 8. Visualisation: box plot
# These plots are optional outputs from the code.
# If the PNG files already exist, running this code will simply overwrite them.

plot_data = [
    below_equal,
    above
]

plt.figure(figsize=(7, 5))
plt.boxplot(
    plot_data,
    tick_labels=[
        "Below/equal-median",
        "Above-median"
    ]
)
plt.ylabel("Shots on target per 90 minutes")
plt.xlabel("Tournament possession group")
plt.title("Shots on target per 90 by tournament possession group")
plt.tight_layout()
plt.savefig("possession_group_boxplot.png", dpi=200)
plt.show()

# 9. Secondary visualisation: scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(
    sample["Possession_Percentage"],
    sample["SoT_per_90"]
)
plt.axvline(
    median_possession,
    linestyle="--",
    label=f"Sample median = {median_possession:.1f}%"
)
plt.xlabel("Tournament possession percentage")
plt.ylabel("Shots on target per 90 minutes")
plt.title("Tournament possession vs shots on target per 90")
plt.legend()
plt.tight_layout()
plt.savefig("possession_vs_sot_scatter.png", dpi=200)
plt.show()
