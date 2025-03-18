import pandas as pd
from sklearn.model_selection import train_test_split  # To split data into training and testing
from sklearn.ensemble import RandomForestRegressor  # The model we will use
from sklearn.metrics import mean_squared_error  # To measure the model's performance
from sklearn.preprocessing import StandardScaler  # To scale the features



data = '/Users/faisalshariff/Desktop/Programming/Datasets/NASCAR Champion History Dataset.csv'

df = pd.read_csv(data)

#print(df.head())
#print(df.columns)

# Filter for years >= 2000
df_recent = df[df['Year'] >= 2000]


# Total wins 
df_wins = df_recent.groupby('Driver')['Wins'].sum().reset_index()

# Number of seasons per driver
df_seasons = df_recent.groupby('Driver')['Year'].nunique().reset_index()
df_seasons.rename(columns={'Year': 'Seasons'}, inplace=True)
races_per_season = 36



# Merge the two dataframes into a new dataframe
df_merged = pd.merge(df_wins, df_seasons, on='Driver')


# Win rate formula : (Total wins / number of seasons) * 100
df_merged['Win Rate'] = (df_merged['Wins'] / (df_merged['Seasons'] * 36)) * 100
df_merged_sorted = df_merged.sort_values(by='Win Rate', ascending=False)

# X features and Y target
X = df_merged_sorted[['Wins', 'Seasons']]
y = df_merged_sorted['Win Rate']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=99)

# Scale the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=99)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Measure the model's performance
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Display predictions sorted by predicted Win Rate
df_test_predictions = pd.DataFrame({'Driver': df_merged['Driver'].iloc[X_test.index], 'Predicted Win Rate': y_pred})
df_test_predictions_sorted = df_test_predictions.sort_values(by='Predicted Win Rate', ascending=False)
print(df_test_predictions_sorted)





