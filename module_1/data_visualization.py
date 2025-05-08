# ====== Data Visualization =====
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Set global plot style
sns.set(style="whitegrid")


# == BAR PLOT ==
# Bar plot of remodel types
df = pd.read_csv("../data/WestRoxbury.csv")
df = df.fillna("None")
remodel_counts = df['REMODEL'].value_counts()
print(remodel_counts)  # break down of each category type in the column

remodel_counts.plot(kind='bar')
plt.title("Number of Homes: Not Remodeled vs. Recently vs. Long Ago")
plt.xlabel("Remodel Type")
plt.ylabel("Number of Homes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# == LINE GRAPH == (COMMENTED OUT BECAUSE YFINANCE IS HAVING ISSUES)
# Download stock data using yfinance
# amzn = yf.download("AMZN", start="2019-01-01")
#
# # Reset index to get 'Date' as a column
# print(amzn.index)
# amzn = amzn.reset_index()  # by default, date is the index with yfinance data, here we reset it so date becomes a normal column rather than index
# print(amzn.index)
#
# # Line plot of adjusted closing price
# plt.figure(figsize=(10, 5))
# sns.lineplot(data=amzn, x="Date", y="Adj Close")
# plt.title("Amazon Stock Price Since 2018")
# plt.xlabel("Date")
# plt.ylabel("Adjusted Closing Price")
# plt.tight_layout()
# plt.show()


# == BOX PLOTS ==
from sklearn.datasets import load_iris

# Load iris data
iris = load_iris(as_frame=True)
iris_df = iris.frame
iris_melted = pd.melt(iris_df, id_vars='target', value_vars=iris.feature_names,
                      var_name='dimension', value_name='centimeters')

# Boxplot without species
plt.figure(figsize=(8, 5))
sns.boxplot(data=iris_melted, x='dimension', y='centimeters')
plt.title("Iris Flower Dimensions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Add species label
iris_melted['Species'] = iris_melted['target'].map(dict(zip(range(3), iris.target_names)))

plt.figure(figsize=(8, 5))
sns.boxplot(data=iris_melted, x='dimension', y='centimeters', hue='Species')
plt.title("Iris Flower Dimensions by Species")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# == HEAT MAP ==
# Using iris data again
corr_matrix = iris_df.iloc[:, :4].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap: Iris Features")
plt.tight_layout()
plt.show()




# == SCATTER PLOT ==# Simulated version since scraping Wikipedia would require BeautifulSoup or requests_html
# For demo, let's assume we have this kind of structure:

data = {
    'County': ['County A', 'County B', 'County C', 'County D'],
    'Population': [50000, 120000, 75000, 30000],
    'PerCapitaIncome': [32000, 41000, 39000, 28000],
    'State': ['New York', 'Massachusetts', 'New York', 'Massachusetts']
}
poptable = pd.DataFrame(data)

# Scatter plot
plt.figure(figsize=(7, 5))
sns.scatterplot(data=poptable, x='Population', y='PerCapitaIncome', hue='State')
plt.title("Income vs Population by MA and NY Counties")
plt.xlabel("Population")
plt.ylabel("Income per Capita")
plt.tight_layout()
plt.show()