import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# show all functions
# df = pd.DataFrame()
# for f in dir(df):
#     print(f)

df = pd.read_csv('autism-adult.csv')
# print(df.head)
# print(df.columns)
# print(df.dtypes)
#
# # Count NaN values
# print(df.isna().sum())  # per column
# print(df.isna().sum().sum())  # per entire df


# Utility functions
def remove_outliers_iqr(df, column, extreme=False):
    """
    Removes outliers from a specific df column using the IQR method.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        column (str): Column to clean.
        extreme (bool): If True, uses 3x IQR (for extreme outliers).
                        If False, uses 1.5x IQR (default statistical rule).

    Returns:
        pd.DataFrame: Filtered DataFrame with outliers removed.
    """
    multiplier = 3.0 if extreme else 1.5

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"Detected {'extreme' if extreme else 'standard'} outliers in '{column}': {outliers[[column]]}")
    # note: also reports row-index which initially confusing

    return df[(df[column] >= lower) & (df[column] <= upper)]


# Problem 1
# 1.1
# print("Unique Values: ", df['age'].unique())  # Identify Problematic Values
df['age'] = pd.to_numeric(df['age'], errors='coerce')  # treat '?' values as NaN
# df = remove_outliers_iqr(df, 'age', extreme=True)  # Remove extreme outliers

age_mean = df['age'].mean()
age_median = df['age'].median()
age_std = df['age'].std()
print(f"Mean Age: {age_mean.round(2)}, Median Age: {age_median.round(2)}, Standard Deviation Age: {age_std.round(2)}")
"""
NOTE: had to convert age from object data type to numeric type. In order to do this, viewed unique values and
spotted a '?' had to treat these values as NaN then run calculations. Also removed extreme outliers using 3X IQR.
"""

# 1.2
quartiles = df['age'].quantile([0.25, 0.5, 0.75])
print("Quartiles: ", quartiles)

# 1.3
sns.boxplot(data=df['age'])
plt.title('Box Plot: Age Distribution')
plt.show()

# 1.4
scaler = MinMaxScaler()
# rescales to 0-1 range
df[['age']] = scaler.fit_transform(df[['age']])  # min-max scaler expects a 2d input, so wrap in double brackets
# print(df['age'].tolist())
print("Age 7th Observation:", df['age'].iloc[6])  # 7th observation (0 indexed), iloc is more explicit about position

# 1.5
print("Country of Residence Mode: ", df['country_of_res'].mode())

# 1.6
print("Unique Ethnicity Values: ", df['ethnicity'].unique())

# # Standardize text formatting
# df['ethnicity'] = df['ethnicity'].astype(str).str.strip().str.replace("'", "").str.title()

# Replace true missing values with "Unknown"
df['ethnicity'] = df['ethnicity'].replace('?', 'Unknown')

# # Normalize "Others" to "Other"
# df['ethnicity'] = df['ethnicity'].replace(['Others', 'Others ', 'others'], 'Other')
"""
NOTE: "?" were interpreted as missing and replaced with 'Unknown', variants of "Others" were standardized to "Other".
Importantly, did not mark unknown as others, because others contains more information than unknown. It implies it is not
one of the listed categories. 
"""

# 1.7
ethnicity_counts = df['ethnicity'].value_counts()
print(ethnicity_counts)  # breakdown of each category type in the column

ethnicity_counts.plot(kind='bar')
plt.title("Reported Ethnicities")
plt.xlabel("Ethnicity Type")
plt.ylabel("Number of Reports")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
NOTE: did not further group, but could be appropriate. However, very subjective.
Example: lumping Turkish with "Other" or with "Middle Eastern"
"""

# 1.8
df = pd.get_dummies(df, columns=['gender'], drop_first=True)
print(df['gender_m'].tail(10))

# 1.9
# discrete vs continuous
# Discrete:  A1 to A10 score, ethnicity, jaundice, autism, country_of_res, used_app_before, relation, class/ASD
# Continuous: age

# 1.10
numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=['number', 'bool']).columns.tolist()

print("Numeric:", numeric_cols)
print("Non-numeric:", non_numeric_cols)
# NOTE: gender was non-numeric prior to encoding, and the numeric after (boolean)

# 1.11
print(df.head(4).to_string())  # set to_string to display full width results

# Problem 2
df = pd.read_csv('correlation.csv')
# print(df.head)
# print(df.columns)
# print(df.dtypes)

# # Count NaN values
# print(df.isna().sum())  # per column
# print(df.isna().sum().sum())  # per entire df

# 2.1 Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="A1", y="A5")
plt.title("Scatterplot of A1 vs A5")
plt.xlabel("A1")
plt.ylabel("A5")
plt.grid(True)
plt.show()

# 2.2 Correlation matrix
corr_matrix = df.corr()
print("Correlation Matrix:\n", corr_matrix)

# Heatmap Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# 2.3
# Strongest correlation A3 vs A2 (.4645), followed closely by A1 vs A5 (.4524)

# 2.4
# Z-score normalization
df = df.apply(zscore)

# 2.5
normalized_corr_matrix = df.corr()
print("\nCorrelation Matrix after Normalization:\n", normalized_corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(normalized_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
plt.title("Correlation Matrix Heatmap (After Z-score Normalization)")
plt.tight_layout()
plt.show()
# Matrix is the same as before, z-score normalization doesn't change relative positioning of data, just rescales
