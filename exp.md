Sure, I'll include the code lines and provide some relevant links for further reading. Here's the step-by-step explanation with code lines:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
```

1. First, we import the necessary libraries:
   - `pandas` is used for data manipulation and analysis. [Documentation](https://pandas.pydata.org/docs/)
   - `numpy` is used for numerical operations. [Documentation](https://numpy.org/doc/)
   - `sklearn` (scikit-learn) is a machine learning library that provides tools for data preprocessing, model selection, and evaluation. [Documentation](https://scikit-learn.org/stable/user_guide.html)
   - `matplotlib` and `seaborn` are data visualization libraries that help us create plots and graphs. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html), [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)

```python
df = pd.read_csv('data_concrete.csv')
df["Number of Days"] = df["Number of Days"].astype(str)
df.info()
```

2. Next, we read the data from a CSV file named `data_concrete.csv` and store it in a pandas DataFrame called `df`.
3. We convert the "Number of Days" column from numeric to string data type using `df["Number of Days"] = df["Number of Days"].astype(str)`.
4. We print information about the DataFrame using `df.info()`.

```python
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
plt.suptitle('Strength Measures by Concrete Grade')

strengths = ['Comp_Strength (N/mm2)', 'Flex_strenght(N/mm2)', 'Splitting Strength(N/mm2)']
titles = ['Compressive Strength', 'Flexural Strength', 'Splitting Strength']

for ax, strength, title in zip(axs, strengths, titles):
  sns.boxplot(x='Grade', y=strength, data=df, ax=ax)
  ax.set_title(title)
  ax.set_xlabel('Grade')
  ax.set_ylabel('Strength (N/mm2)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

5. We create a figure with three subplots, one for each strength measure (compressive, flexural, and splitting), using `fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)`. The `sharey=True` argument ensures that the y-axis scales are the same across all subplots.
6. We loop through the strength measures, creating a boxplot for each one using `sns.boxplot()`. The boxplots show the distribution of each strength measure by concrete grade.

```python
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.suptitle('Strength Measures vs. Glass Fibre (%)')

for ax, strength, title in zip(axs, strengths, titles):
  ax.scatter(df['Glass Fibre (%)'], df[strength])
  ax.set_title(f'{title} vs. Glass Fibre %')
  ax.set_xlabel('Glass Fibre (%)')
  ax.set_ylabel('Strength (N/mm2)')
  # Fit and plot a linear regression line for each strength measure
  sns.regplot(x='Glass Fibre (%)', y=strength, data=df, ax=ax, scatter=False, color='red')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

7. We create another figure with three subplots, again one for each strength measure, using `fig, axs = plt.subplots(1, 3, figsize=(18, 6))`.
8. We loop through the strength measures, creating a scatter plot for each one using `ax.scatter()`. The scatter plots show the relationship between each strength measure and the percentage of glass fiber in the concrete.
9. We fit a linear regression line to each scatter plot using `sns.regplot()`. This line shows the general trend between each strength measure and the percentage of glass fiber.

```python
df[['Comp_Strength (N/mm2)']].plot(kind='line', xticks=range(len(df)), rot=90,figsize=(100,5))
df[['Flex_strenght(N/mm2)']].plot(kind='line', xticks=range(len(df)), rot=90,  figsize=(100,5))
df[['Splitting Strength(N/mm2)']].plot(kind='line', xticks=range(len(df)), rot=90, figsize=(100,5))
```

10. We plot the compressive, flexural, and splitting strengths over the range of data points using `df[['Comp_Strength (N/mm2)']].plot(kind='line', xticks=range(len(df)), rot=90, figsize=(100,5))`. This allows us to visualize the strengths as a continuous line plot.

```python
df_model = df.copy(deep=True)
# One-hot encoding "Grade" and "Number of Days"
encoder = OneHotEncoder()
encode = encoder.fit(df_model[['Grade', 'Number of Days']])
encoded_features = encode.transform(df_model[['Grade', 'Number of Days']]).toarray()

# Normalizing "Glass Fibre (%)"
scaler = MinMaxScaler()
gf_scaled = scaler.fit_transform(df_model[['Glass Fibre (%)']])

# Preparing the features (X) and target (y)
X = np.concatenate([encoded_features, gf_scaled], axis=1)
y_comp = df_model['Comp_Strength (N/mm2)'].values
y_flex = df_model['Flex_strenght(N/mm2)'].values
y_split = df_model['Splitting Strength(N/mm2)'].values
```

11. We create a copy of the DataFrame called `df_model` using `df_model = df.copy(deep=True)`.
12. We one-hot encode the "Grade" and "Number of Days" columns using `encoder = OneHotEncoder()` and `encoded_features = encode.transform(df_model[['Grade', 'Number of Days']]).toarray()`. One-hot encoding is a technique used to convert categorical data into a numerical format that can be used in machine learning models. [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
13. We normalize the "Glass Fibre (%)" column using `scaler = MinMaxScaler()` and `gf_scaled = scaler.fit_transform(df_model[['Glass Fibre (%)']])`. Normalization is a preprocessing step that helps ensure that features with different scales don't have a disproportionate impact on the model. [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
14. We prepare the input features (X) and target variables (y_comp, y_flex, y_split) by concatenating the one-hot encoded features and the normalized glass fiber percentage.

```python
model_comp = LinearRegression()
model_flex = LinearRegression()
model_split = LinearRegression()

model_comp.fit(X, y_comp)
model_flex.fit(X, y_flex)  
model_split.fit(X, y_split)

y_pred_comp = model_comp.predict(X)
y_pred_flex = model_flex.predict(X)
y_pred_split = model_split.predict(X)
```

15. We create three instances of the LinearRegression model from scikit-learn, one for each strength measure: `model_comp`, `model_flex`, and `model_split`. [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
16. We fit each model to the corresponding target variable (compressive, flexural, and splitting strengths) using the input features (X) with `model_comp.fit(X, y_comp)`, `model_flex.fit(X, y_flex)`, and `model_split.fit(X, y_split)`.
17. We make predictions for each strength measure using the fitted models with `y_pred_comp = model_comp.predict(X)`, `y_pred_flex = model_flex.predict(X)`, and `y_pred_split = model_split.predict(X)`.

```python
def predict_strengths(grade, days, percentage_mix):
  # Encode the grade and days using the previously defined encoder
  encoded_grade_days = encode.transform([[grade, days]]).toarray()

  # Normalize the percentage mix
  normalized_percentage_mix = scaler.transform([[percentage_mix]])

  # Prepare the input features
  input_features = np.concatenate([encoded_grade_days, normalized_percentage_mix], axis=1)

  # Predict the strengths using the trained models
  predicted_comp_strength = model_comp.predict(input_features)
  predicted_flex_strength = model_flex.predict(input_features)
  predicted_split_strength = model_split.predict(input_features)

  # Return the predicted strengths
  return print(f'Predicted Compressive Strength: {predicted_comp_strength[0]:.2f} N/mm2\nPredicted Flexural Strength: {predicted_flex_strength[0]:.2f} N/mm2\nPredicted Splitting Strength: {predicted_split_strength[0]:.2f} N/mm2')
```

18. We define a function `predict_strengths(grade, days, percentage_mix)` that takes the grade, number of days, and glass fiber percentage as input. This function:
    - Encodes the input grade and days using the previously defined encoder.
    - Normalizes the input glass fiber percentage using the previously defined scaler.
    - Prepares the input features by concatenating the encoded grade and days with the normalized glass fiber percentage.
    - Predicts the compressive, flexural, and splitting strengths using the trained models.
    - Prints the predicted strengths.

The code covers data preprocessing, exploratory data analysis, model training, and prediction. It provides a way to visualize the data and relationships between variables, as well as predict the concrete strengths based on input features.

Here are some relevant links for further reading:

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)
- [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)