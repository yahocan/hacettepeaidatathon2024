# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("ggplot")  # just set this once

# %%
# quick mapping of files
files = {
    "News": "Newsletter_Interaction_Data.csv",
    "Bugs": "Product_Bug_Task_Data.csv",
    "Regions": "RegionAndVertical_Data.csv",
    "Status": "StatusAndLevel_Data.csv",
    "CSAT": "CSAT_Survey_Data.csv",
    "Age": "Customer_Age_Data.csv",
    "MRR": "Customer_MRR_Data.csv",
    "Rev": "Customer_Revenue_Data.csv",
    "Tickets": "Help_Ticket_Data.csv",
}

# %%
# load everything at once
dfs = {}
for k, v in files.items():
    try:
        dfs[k] = pd.read_csv(f"archive/{v}")
    except:
        print(f"couldn't load {v}")

# %%
# fix ID column mismatch
dfs["Age"].rename(columns={"CRM ID": "Customer ID"}, inplace=True)


# %%
# clean money stuff
def fix_money(x):
    if isinstance(x, str):
        return float(x.replace("$", "").replace(",", ""))
    return x


# apply to money cols
dfs["MRR"]["MRR"] = dfs["MRR"]["MRR"].apply(fix_money)
dfs["Rev"]["Total Revenue"] = dfs["Rev"]["Total Revenue"].apply(fix_money)

# %%
# check what we got
for name, df in dfs.items():
    print(f"{name}: {df['Customer ID'].nunique()} customers, {len(df)} rows")

# %%
print(dfs["CSAT"].columns)

# %%
# aggregate data - this is messy but works
dfs["Bugs"] = (
    dfs["Bugs"].groupby("Customer ID", as_index=False)["Product Bug Task Count"].sum()
)

csat_cols = [
    "How likely are you to recommend insider to a friend or colleague ",
    "How would you rate the value you gain from our company",
    "Please rate the overall quality of our products",
    "Please rate the usability of the panel",
]
dfs["CSAT"] = dfs["CSAT"].groupby("Customer ID", as_index=False)[csat_cols].mean()

# mode for categorical stuff
mode_agg = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
dfs["Regions"] = dfs["Regions"].groupby("Customer ID").agg(mode_agg).reset_index()
dfs["Status"] = dfs["Status"].groupby("Customer ID").agg(mode_agg).reset_index()

# %%
# merge everything - start with age df
df = dfs["Age"]

# add the rest one by one
for k, v in dfs.items():
    if k != "Age":  # skip the first one
        df = df.merge(v, on="Customer ID", how="left")

# %%
# quick size check
print(f"Got {len(df)} rows, {df['Customer ID'].nunique()} unique customers")

for k, v in dfs.items():
    print(f"{k}: {v['Customer ID'].nunique()} IDs, {len(v)} rows")

# %%
# see what's missing
miss = df.isnull().sum()
miss_pct = (miss / len(df)) * 100

miss_df = pd.DataFrame({"Missing": miss, "Pct": miss_pct})
miss_df = miss_df[miss_df["Missing"] > 0]

print("\nMissing data:")
print(miss_df)

# %%
# fix missing vals - quick and dirty
df["Company Newsletter Interaction Count"] = df[
    "Company Newsletter Interaction Count"
].fillna(0)
df["Product Bug Task Count"] = df["Product Bug Task Count"].fillna(0)

# fill numeric with median
nums = df.select_dtypes(include=[np.number]).columns
for col in nums:
    df[col] = df[col].fillna(df[col].median())

# zero $ fields explicitly
df["MRR"] = df["MRR"].fillna(0)
df["Total Revenue"] = df["Total Revenue"].fillna(0)
df["Help Ticket Count"] = df["Help Ticket Count"].fillna(0)
df["Help Ticket Lead Time (hours)"] = df["Help Ticket Lead Time (hours)"].fillna(0)

# mode for text fields
cats = ["Region", "Vertical", "Subvertical", "Customer Level"]
for col in cats:
    df[col] = df[col].fillna(df[col].mode()[0])

# %%
print(df.isnull().sum())

# %% [markdown]
# Data cleaned up

# %%
print(df.describe())

# %%
print(df.dtypes)

# %%
# histograms - get all numeric cols
num_cols = [
    "Customer Age (Months)",
    "Company Newsletter Interaction Count",
    "Product Bug Task Count",
    "MRR",
    "Total Revenue",
    "Help Ticket Count",
    "Help Ticket Lead Time (hours)",
]

plt.figure(figsize=(12, 8))
df[num_cols].hist(bins=30, figsize=(12, 8), layout=(3, 3), color="darkblue", alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# boxplots for $ cols and support metrics
money_cols = [
    "MRR",
    "Total Revenue",
    "Help Ticket Count",
    "Help Ticket Lead Time (hours)",
]

fig = plt.figure(figsize=(12, 8))
for i, c in enumerate(money_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[c], color="crimson")
    plt.title(f"{c}")

plt.tight_layout()
plt.show()

# %%
# find outlier caps at 99%
caps = {}
money_cols = [
    "MRR",
    "Total Revenue",
    "Help Ticket Count",
    "Help Ticket Lead Time (hours)",
]

for c in money_cols:
    q99 = df[c].quantile(0.99)
    caps[c] = q99
    print(f"{c} 99% cap: {q99}")

# %%
# cap outliers
for c, cap in caps.items():
    df[c] = df[c].apply(lambda x: min(x, cap))

# %%
print(df[money_cols].describe())

# %%
# replot boxplots post-capping
fig = plt.figure(figsize=(12, 8))
for i, c in enumerate(money_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[c], color="forestgreen")
    plt.title(f"{c} (capped)")

plt.tight_layout()
plt.show()

# %%
# corr matrix - most interesting cols
corr = df[
    [
        "MRR",
        "Total Revenue",
        "Help Ticket Count",
        "Help Ticket Lead Time (hours)",
        "How likely are you to recommend insider to a friend or colleague ",
        "How would you rate the value you gain from our company",
    ]
].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="RdBu_r", fmt=".2f", linewidths=0.5)
plt.show()

# %%
# group MRR into buckets
df["MRR_grp"] = pd.qcut(df["MRR"], 4, labels=["Low", "Mid-Low", "Mid-High", "High"])

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="MRR_grp",
    y="How likely are you to recommend insider to a friend or colleague ",
    data=df,
    hue="MRR_grp",
    legend=False,
    palette="viridis",
)
plt.title("NPS by MRR Group")
plt.xlabel("MRR Group")
plt.ylabel("NPS Score")
plt.show()

# %%
# status vs NPS
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Status",
    y="How likely are you to recommend insider to a friend or colleague ",
    data=df,
    palette="magma",
    hue="Status",
    legend=False,
)
plt.title("NPS: Churned vs Active")
plt.xlabel("Status")
plt.ylabel("NPS Score")
plt.show()


# %%
# NPS buckets
def nps_group(s):
    if s <= 6:
        return "Low"
    if s <= 8:
        return "Neutral"
    return "High"


df["NPS_grp"] = df[
    "How likely are you to recommend insider to a friend or colleague "
].apply(nps_group)

plt.figure(figsize=(9, 5))
sns.countplot(x="NPS_grp", hue="Status", data=df, palette="viridis")
plt.title("Customer Status by NPS Group")
plt.xlabel("NPS Group")
plt.ylabel("Count")
plt.legend(title="Status")
plt.show()

# %% [markdown]
# Weird - lots of high NPS customers still churned!

# %%
# high NPS but churned - why?
happy_churns = df[(df["NPS_grp"] == "High") & (df["Status"] == "Churn")]

plt.figure(figsize=(10, 6))
sns.boxplot(y=happy_churns["MRR"], color="#ff5533")
plt.title("MRR of Churned High-NPS Customers")
plt.ylabel("MRR ($)")
plt.yscale("log")
plt.show()

# %%
# churn by segment
plt.figure(figsize=(10, 6))
sns.countplot(x="Customer Level", hue="Status", data=df, palette="mako")
plt.title("Churn by Customer Tier")
plt.xlabel("Segment")
plt.ylabel("Count")
plt.legend(title="Status")
plt.xticks(rotation=45)
plt.show()

# %%
# long-tail churns - NPS
tail_churns = df[(df["Customer Level"] == "Long-tail") & (df["Status"] == "Churn")]

plt.figure(figsize=(10, 6))
sns.boxplot(
    y=tail_churns["How likely are you to recommend insider to a friend or colleague "],
    color="#dd4477",
)
plt.title("NPS: Churned Long-tail Customers")
plt.ylabel("NPS Score")
plt.show()

# %%
# long-tail churns - MRR
tail_churns = df[(df["Customer Level"] == "Long-tail") & (df["Status"] == "Churn")]

plt.figure(figsize=(10, 6))
sns.boxplot(y=tail_churns["MRR"], color="#dd4477")
plt.title("MRR: Churned Long-tail Customers")
plt.ylabel("MRR ($)")
plt.yscale("log")
plt.show()

# %%
# ticket counts for churns
churns = df[df["Status"] == "Churn"]

plt.figure(figsize=(10, 6))
sns.boxplot(y=churns["Help Ticket Count"], color="#9933aa")
plt.title("Support Tickets: Churned Customers")
plt.ylabel("Ticket Count")
plt.yscale("log")
plt.show()

# %%
# resolution time for churns
churns = df[df["Status"] == "Churn"]

plt.figure(figsize=(10, 6))
sns.boxplot(y=churns["Help Ticket Lead Time (hours)"], color="#9933aa")
plt.title("Support Resolution Time: Churned Customers")
plt.ylabel("Hours")
plt.yscale("log")
plt.show()

# %% [markdown]
# Customer support times look really bad - might be a major churn driver

# %%
# support ticket scatter
churns = df[df["Status"] == "Churn"]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=churns["Help Ticket Count"],
    y=churns["Help Ticket Lead Time (hours)"],
    alpha=0.6,
    color="#663399",
)
plt.title("Support Tickets vs Resolution Time (Churned)")
plt.xlabel("Ticket Count")
plt.ylabel("Resolution Hours")
plt.xscale("log")
plt.yscale("log")
plt.show()

# %%
# bugs vs resolution time
plt.figure(figsize=(10, 6))
sns.boxplot(
    x=df["Product Bug Task Count"],
    y=df["Help Ticket Lead Time (hours)"],
    hue=df["Product Bug Task Count"],
    legend=False,
    palette="flare",
)

plt.title("Resolution Time vs Bug Tasks")
plt.xlabel("Bug Tasks")
plt.ylabel("Resolution Hours")
plt.show()
