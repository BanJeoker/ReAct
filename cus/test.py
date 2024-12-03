import pandas as pd

# Initialize the final DataFrame
df_final = pd.DataFrame(columns=df.columns)  # Using the same columns as the original DataFrame

# Iterate through the columns in chunks of 6
num_columns = df.shape[1]  # Total number of columns including CUSIP
num_chunks = num_columns // 6  # Number of sets of 6 columns

for chunk in range(num_chunks):
    # Get the corresponding set of 6 columns
    cols = df.columns[chunk * 6 : (chunk + 1) * 6]
    
    # Create a new DataFrame with these 6 columns
    df_split = df[cols]
    
    # Append this new DataFrame to df_final
    df_final = pd.concat([df_final, df_split], ignore_index=True)

# Display the reshaped DataFrame
import ace_tools as tools; tools.display_dataframe_to_user(name="Reshaped Data", dataframe=df_final)





import pandas as pd

# Example DataFrame with datetime columns
df = pd.DataFrame({
    'date_column1': ['2024-01-01', '2024-02-01', '2024-03-01'],
    'date_column2': ['2023-12-31', '2024-02-02', '2024-02-28']
})

# Convert the columns to datetime type if not already in datetime64
df['date_column1'] = pd.to_datetime(df['date_column1'])
df['date_column2'] = pd.to_datetime(df['date_column2'])

# Check if date_column1 is later than date_column2
df['is_later'] = df['date_column1'] > df['date_column2']

print(df)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example DataFrame
df = pd.DataFrame({
    'col1': [1, 2, 3, 4, 5],
    'col2': [5, 4, 3, 2, 1],
    'col3': [10, 20, 30, 40, 50],
    'target': [5, 10, 15, 20, 25]
})

# Calculate the correlation matrix
correlation = df[['col1', 'col2', 'col3', 'target']].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)

# Display the plot
plt.show()
