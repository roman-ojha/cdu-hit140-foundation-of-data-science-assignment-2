# Task Question:
# Is the mean tournament passing accuracy significantly higher for teams that advanced to the knockout stage than for teams eliminated in the group stage?

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Configuring file input and output paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/world_cup_2026_team_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Input file: {INPUT_FILE}")


# Some constants values
SEED = 2026
SAMPLE_PER_GROUP = 16
ALPHA = 0.05

# Some of the required columns from the csv file.
REQUIRED_COLUMNS = [
    "Rank",
    "Team",
    "passing_accuracy",
    "pass_completed",
    "advanced",
]

# Load dataset
df = pd.read_csv(INPUT_FILE)

# converting data types and removing whitespace from the data
df["Team"] = df["Team"].astype(str).str.strip()
df["passing_accuracy"] = pd.to_numeric(
    df["passing_accuracy"], errors="coerce"
)
df["pass_completed"] = pd.to_numeric(
    df["pass_completed"], errors="coerce"
)
df["advanced"] = pd.to_numeric(df["advanced"], errors="coerce")

# Removing all the invalid rows with missing values
df = df.dropna(
    subset=["Team", "passing_accuracy", "pass_completed", "advanced"]
).copy()

# Checking binary grouping variable
valid_advanced_values = set(df["advanced"].unique())
if not valid_advanced_values.issubset({0, 1}):
    raise ValueError(
        f"'advanced' must contain only 0 and 1. "
        f"Found: {sorted(valid_advanced_values)}"
    )

# Checking exactly one row per team
if df["Team"].duplicated().any():
    duplicates = df.loc[df["Team"].duplicated(), "Team"].tolist()
    raise ValueError(f"Duplicate team rows found: {duplicates}")

# Adding readable group labels
df["group"] = df["advanced"].map(
    {1: "Advanced to knockout", 0: "Eliminated in group stage"}
)

# Required World Cup structure for this dataset
advanced_count = int((df["advanced"] == 1).sum())
eliminated_count = int((df["advanced"] == 0).sum())

# Checking whether the dataset contain the expected number of teams or not
if len(df) != 48:
    print(
        f"Please provide 48 teams in dataset, we only found {len(df)}.")

# Checking whether the dataset contain 32 teams that advanced to the knockout stage and 16 teams that eliminate in the group stage.
if advanced_count != 32 or eliminated_count != 16:
    print(
        "Expected 32 advanced and 16 eliminated teams, "
        f"but found {advanced_count} advanced and {eliminated_count} eliminated."
    )


# Calculating descriptive statistics for the full teams
def descriptive_table(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("group")["passing_accuracy"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .assign(
            range=lambda x: x["maximum"] - x["minimum"]
        )
        .reset_index()
    )


full_descriptive = descriptive_table(df)
full_descriptive.to_csv(
    OUTPUT_DIR / "full_descriptive_statistics.csv", index=False
)


# Splitting the dataset into advanced and eliminated groups
advanced_df = df[df["advanced"] == 1].copy()
eliminated_df = df[df["advanced"] == 0].copy()

if len(advanced_df) < SAMPLE_PER_GROUP:
    raise ValueError("Not enough advanced teams for the requested sample")

if len(eliminated_df) < SAMPLE_PER_GROUP:
    raise ValueError("Not enough eliminated teams for the requested sample")

sample_advanced = advanced_df.sample(
    n=SAMPLE_PER_GROUP, random_state=SEED
)
sample_eliminated = eliminated_df.sample(
    n=SAMPLE_PER_GROUP, random_state=SEED
)

sample = pd.concat(
    [sample_advanced, sample_eliminated],
    ignore_index=True
).sample(frac=1, random_state=SEED).reset_index(drop=True)

sample.to_csv(OUTPUT_DIR / "analysis_sample_32_teams.csv", index=False)

# Splitting sampled passing accuracy values into two groups for analysis
x_advanced = sample.loc[
    sample["advanced"] == 1, "passing_accuracy"
].to_numpy()

x_eliminated = sample.loc[
    sample["advanced"] == 0, "passing_accuracy"
].to_numpy()

n1 = len(x_advanced)
n2 = len(x_eliminated)

mean1 = np.mean(x_advanced)
mean2 = np.mean(x_eliminated)
std1 = np.std(x_advanced, ddof=1)
std2 = np.std(x_eliminated, ddof=1)
difference = mean1 - mean2

# This function will calculate the mean and 95% confidence interval


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    low, high = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=se,
    )
    return float(mean), float(low), float(high)


adv_mean, adv_ci_low, adv_ci_high = mean_ci(x_advanced)
elim_mean, elim_ci_low, elim_ci_high = mean_ci(x_eliminated)


var1 = np.var(x_advanced, ddof=1)
var2 = np.var(x_eliminated, ddof=1)

se_difference = np.sqrt(
    var1 / n1 + var2 / n2
)

# Calculating the degrees of freedom for Welch's t-test
welch_df = (
    (var1 / n1 + var2 / n2) ** 2
    / (
        (var1 / n1) ** 2 / (n1 - 1)
        + (var2 / n2) ** 2 / (n2 - 1)
    )
)

t_critical = stats.t.ppf(1 - ALPHA / 2, df=welch_df)

difference_ci_low = difference - t_critical * se_difference
difference_ci_high = difference + t_critical * se_difference

# we will use Shapiro-Wilk test for checking normality for passing accuracy among advanced team
shapiro_adv = stats.shapiro(x_advanced)
# also Shapiro test for checking normality for passing accuracy among eliminated team
shapiro_elim = stats.shapiro(x_eliminated)

# We will do Levene's test to check two groups variance
levene_result = stats.levene(
    x_advanced,
    x_eliminated,
    center="median",
)

# Performing two sample t-test
#     H0: mu_advanced <= mu_eliminated
#     H1: mu_advanced > mu_eliminated
ttest_result = stats.ttest_ind(
    x_advanced,
    x_eliminated,
    equal_var=False,
    alternative="greater",
)

# Calculating pooled standard deviation for Cohen's d
pooled_sd = np.sqrt(
    (
        (n1 - 1) * var1
        + (n2 - 1) * var2
    )
    / (n1 + n2 - 2)
)

cohens_d = difference / pooled_sd

# This function will return the outliers based on the IQR rule


def iqr_outliers(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return values[(values < lower) | (values > upper)], lower, upper


adv_outliers, adv_lower, adv_upper = iqr_outliers(x_advanced)
elim_outliers, elim_lower, elim_upper = iqr_outliers(x_eliminated)

# Creating a DataFrame to store the inferential results
results = pd.DataFrame(
    [
        {
            "sample_size_advanced": n1,
            "sample_size_eliminated": n2,
            "mean_advanced": mean1,
            "mean_eliminated": mean2,
            "mean_difference": difference,
            "advanced_mean_95CI_low": adv_ci_low,
            "advanced_mean_95CI_high": adv_ci_high,
            "eliminated_mean_95CI_low": elim_ci_low,
            "eliminated_mean_95CI_high": elim_ci_high,
            "difference_95CI_low": difference_ci_low,
            "difference_95CI_high": difference_ci_high,
            "welch_t_statistic": ttest_result.statistic,
            "welch_df": ttest_result.df,
            "one_sided_p_value": ttest_result.pvalue,
            "cohens_d": cohens_d,
            "shapiro_p_advanced": shapiro_adv.pvalue,
            "shapiro_p_eliminated": shapiro_elim.pvalue,
            "levene_p_value": levene_result.pvalue,
            "alpha": ALPHA,
        }
    ]
)

# Saving the inferential results to a CSV file
results.to_csv(
    OUTPUT_DIR / "inferential_results.csv",
    index=False
)

# ---------------------------------------------------------------------
# 14. Print results to terminal
# ---------------------------------------------------------------------

# Printing all the results to the terminal
print("FIFA World Cup 2026 - Passing Accuracy Analysis ====================")

print("\nData Validation")
print(f"Valid team rows: {len(df)}")
print(f"Advanced teams: {advanced_count}")
print(f"Eliminated teams: {eliminated_count}")

print("\nDataset Descriptive statistics")
print(full_descriptive.to_string(index=False))

print("\nStratified Random Sample of 32 Teams")
print(sample[["Team", "passing_accuracy",
      "advanced", "group"]].to_string(index=False))

print("\nSample Descriptive statistics")
print(f"Advanced mean:  {mean1}%")
print(f"Eliminated mean: {mean2}%")
print(f"Mean difference: {difference} percentage points")

print("\n95% Confidence Intervals")
print(
    f"Advanced mean:   {adv_mean}% "
    f"({adv_ci_low}, {adv_ci_high})"
)
print(
    f"Eliminated mean: {elim_mean}% "
    f"({elim_ci_low}, {elim_ci_high})"
)
print(
    f"Mean difference: {difference} percentage points "
    f"({difference_ci_low}, {difference_ci_high})"
)

print("\nAssumption check")
print(
    f"Shapiro-Wilk, advanced: p = {shapiro_adv.pvalue}"
)
print(
    f"Shapiro-Wilk, eliminated: p = {shapiro_elim.pvalue}"
)
print(
    f"Levene's test: p = {levene_result.pvalue}"
)
print(
    f"IQR outliers, advanced: {len(adv_outliers)}"
)
print(
    f"IQR outliers, eliminated: {len(elim_outliers)}"
)

print("\nTwo Sample T-test")
print(
    "H0: mean passing accuracy (advanced) <= "
    "mean passing accuracy (eliminated)"
)
print(
    "H1: mean passing accuracy (advanced) > "
    "mean passing accuracy (eliminated)"
)

print(f"t = {ttest_result.statistic}")
print(f"df = {ttest_result.df}")
print(f"one-sided p-value = {ttest_result.pvalue}")
print(f"Cohen's d = {cohens_d}")

if ttest_result.pvalue < ALPHA:
    print(
        "\nDecision: Reject H0 at alpha = 0.05.\nFinal Verdict: The sample shows that there is significant evidence that advanced team have higher passing accuracy"
    )
else:
    print(
        "\nDecision: Fail to reject H0 at alpha = 0.05.\nFinal Verdict: The sample doesn't provide enough evidence that shows the advance teams have higher mean passing accuracy"
    )

# Creating a boxplot to visualize the passing accuracy distributions for eliminated and advanced teams
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [x_eliminated, x_advanced],
    tick_labels=["Eliminated", "Advanced"],
)
ax.set_title("Passing Accuracy by Tournament Outcome")
ax.set_ylabel("Passing accuracy (%)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_boxplot_passing_accuracy.png", dpi=300)
plt.close(fig)

# Mean + 95% CI
fig, ax = plt.subplots(figsize=(8, 5))
groups = ["Eliminated", "Advanced"]
means = [elim_mean, adv_mean]
errors_lower = [
    elim_mean - elim_ci_low,
    adv_mean - adv_ci_low,
]
errors_upper = [
    elim_ci_high - elim_mean,
    adv_ci_high - adv_mean,
]

ax.errorbar(
    groups,
    means,
    yerr=[errors_lower, errors_upper],
    fmt="o",
    capsize=6,
    markersize=7,
)
ax.set_title("Mean Passing Accuracy with 95% Confidence Intervals")
ax.set_ylabel("Mean passing accuracy (%)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_mean_ci_passing_accuracy.png", dpi=300)
plt.close(fig)

# Dot plot showing the sampled observations
fig, ax = plt.subplots(figsize=(8, 5))
rng = np.random.default_rng(SEED)

x1 = rng.normal(1, 0.035, size=n2)
x2 = rng.normal(2, 0.035, size=n1)

ax.scatter(x1, x_eliminated, alpha=0.8)
ax.scatter(x2, x_advanced, alpha=0.8)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Eliminated", "Advanced"])
ax.set_title("Sampled Passing Accuracy Values")
ax.set_ylabel("Passing accuracy (%)")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_sample_distribution.png", dpi=300)
plt.close(fig)

# Using all 48 teams to check if the results are consistent with the sample of 32 teams
all_advanced = advanced_df["passing_accuracy"].to_numpy()
all_eliminated = eliminated_df["passing_accuracy"].to_numpy()

all_ttest = stats.ttest_ind(
    all_advanced,
    all_eliminated,
    equal_var=False,
    alternative="greater",
)

all_difference = (
    all_advanced.mean() - all_eliminated.mean()
)

sensitivity = pd.DataFrame(
    [
        {
            "advanced_n": len(all_advanced),
            "eliminated_n": len(all_eliminated),
            "advanced_mean": all_advanced.mean(),
            "eliminated_mean": all_eliminated.mean(),
            "mean_difference": all_difference,
            "welch_t": all_ttest.statistic,
            "welch_df": all_ttest.df,
            "one_sided_p": all_ttest.pvalue,
        }
    ]
)

sensitivity.to_csv(
    OUTPUT_DIR / "sensitivity_all_48_teams.csv",
    index=False
)

print("\nSensitivity Analysis - All 48 Teams")
print(sensitivity.to_string(index=False))

print("\nFiles written to:", OUTPUT_DIR)
