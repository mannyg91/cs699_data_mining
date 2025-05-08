import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../data/WestRoxbury.csv")

# Print first 10 rows
print(df.head(10))

# View dimensions
print(df.shape)

# View column names
print(df.columns)

# View data types of each column
print(df.dtypes)

# Quick summary
print(df.info())

# Summary stats
print(df.describe())

# Count NaN values
print(df.isna().sum())  # per column
print(df.isna().sum().sum())  # per entire df

# Fill NaN with 0 (or other value)
df = df.fillna("None")
print(df.head(10))
print(df.isna().sum().sum())  # per entire df

# Drop specific columns
df = df.drop(columns=['TAX'])  # dropping TAX as likely derived from home value

# Convert three-level categorical feature REMODEL into two dummies
# First, can convert to category datatype which is more efficient for ML
df['REMODEL'] = df['REMODEL'].astype('category')
print(df.dtypes['REMODEL'])  # confirm new datatype

# Dummy encoding for REMODEL column. Three possible values, so splits into two, treating the remaining as the reference column not shown
# Will drop the first one in alphabetical order. "None" in this case. Doesn't matter which one it drops, is arbitrary
df = pd.get_dummies(df, columns=['REMODEL'], drop_first=True)
print(df.columns)  # confirm new dummy variables (appends the category name to the original column name "REMODEL_old" for example

# split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.4, random_state=74551)  # random state seed set for reproducibility
# results in 60% train, 40% test

# separate features and target
X_train = train_df.drop(columns=['TOTAL.VALUE'])
y_train = train_df['TOTAL.VALUE']

X_test = test_df.drop(columns=['TOTAL.VALUE'])
y_test = test_df['TOTAL.VALUE']

# create ML model
lr = LinearRegression()
lr.fit(X_train, y_train) # pass in X and Y training data

# make predictions and evaluate
y_pred = lr.predict(X_test)
residuals = y_test - y_pred
print(residuals)

test_metrics = {
    'ME': round(residuals.mean(), 5),
    'RMSE': mean_squared_error(y_test, y_pred, squared=False),
    'MAE': mean_absolute_error(y_test, y_pred)
}

print(test_metrics)

# predict on new data
new_data = pd.DataFrame({
    'LOT.SQFT': [4200, 6444, 5035],
    'YR.BUILT': [1960, 1940, 1925],
    'GROSS.AREA': [2670, 2886, 3264],
    'LIVING.AREA': [1710, 1474, 1523],
    'FLOORS': [2, 1.5, 1],
    'ROOMS': [10, 6, 6],
    'BEDROOMS': [4, 3, 2],
    'FULL.BATH': [1, 1, 1],
    'HALF.BATH': [1, 1, 0],
    'KITCHEN': [1, 1, 1],
    'FIREPLACE': [1, 1, 0],
    'REMODEL_Old': [0, 0, 0],
    'REMODEL_Recent': [0, 0, 1]
})

predictions = lr.predict(new_data)
print(predictions)


