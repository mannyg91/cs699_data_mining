import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Question 2 - calculate mean and standard deviation of single column of data
df = pd.DataFrame({'v1': [35, 27, 14, 76, 84, 95]})  # define single column into df

print(df)
print('mean', df.mean())
print('stdev', df.std())


# Question 5 - rescale values using min-max of single column. Use interval of [1,100]
# which value is the closest to the re-scaled value of 55? (multiple-choice)
df = pd.DataFrame({'v': [10, 45, 21, 137, 72, 55, 63, 32, 40, 83]})

# Initialize scaler for range [1, 100]
scaler = MinMaxScaler(feature_range=(1, 100))

# Fit and transform the 'v' column
df['v_scaled'] = scaler.fit_transform(df[['v']])

# Extract the rescaled value of 55
rescaled_55 = df.loc[df['v'] == 55, 'v_scaled'].values[0]

print(f"Rescaled value of 55: {rescaled_55:.2f}")


# alternative: simple arithmetic
v = [10, 45, 21, 137, 72, 55, 63, 32, 40, 83]
value_to_rescale = 55

# Min-max normalization to [1, 100]
min_v = min(v)
max_v = max(v)
scaled_value = 1 + ((value_to_rescale - min_v) * (100 - 1)) / (max_v - min_v)

print(f"Rescaled value of 55: {scaled_value:.2f}")


# Question 6 - identifying outliers given quartiles
# Suppose you have a numeric attribute with Q1 = 125 and Q3 = 165.
# Which of the following attribute values are outliers? Use the IQR method. Choose all that apply.

Q1 = 125
Q3 = 165
IQR = Q3-Q1

# find outlier boundaries (1.5x)
print('Lower boundary (1.5x)', Q1 - 1.5 * IQR)
print('Upper boundary (1.5x)', Q3 + 1.5 * IQR)

# Note: book suggests using the >3 stdevs method, but this is a fundamendally different method, not related to IQR.


"""
Question 11
True or False? If two numeric attributes are independent of each other, then the Pearson’s correlation coefficient of thetwo attributes should theoretically be 0 (and will be 0 if you collect infinite data).

Note: I could be mistaken, but I don't remember this being brought up any where in our readings or module.
Question asks for theoretical idea, but in practice I think you would rarely ever get a correlation coefficient of 0?
But, with a low score you would still consider them practically independent? Student might conflate perfect independence with practical independence due to the framing of the question.
"""


"""
Question 12
According to your textbook, which of the following contribute to the need to understand and apply ethical practices in machine learning?Select all that apply.

I left out "The questionable quality of machine learning courses, software, and web sites." because this doesn't appear to be a direct ethical concern, or one in our textbook, but I can see how it could lead to direct ethical concerns.
There's likely a correlation that might reasonably sway some students to select this option. Furthermore, because it mentions "According tyo your textbook.." they authority now lies with the book rather than the general concept..



"""




