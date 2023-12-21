8. establish an sqlite database connection perform crud operations on a table and display results
import sqlite3

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('example.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create a table (if it doesn't exist)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER
    )
''')

# Insert some data
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('John Doe', 25))
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Jane Doe', 30))

# Commit the changes
conn.commit()

# Read data from the table
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

# Display the results
print("Users:")
for row in rows:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}")

# Update data
cursor.execute("UPDATE users SET age=? WHERE name=?", (35, 'John Doe'))
conn.commit()

# Read and display updated data
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

print("\nUpdated Users:")
for row in rows:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}")

# Delete data
cursor.execute("DELETE FROM users WHERE name=?", ('Jane Doe',))
conn.commit()

# Read and display the remaining data
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

print("\nRemaining Users:")
for row in rows:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}")

# Close the connection
conn.close()
output:
Users:
ID: 1, Name: John Doe, Age: 25
ID: 2, Name: Jane Doe, Age: 30

Updated Users:
ID: 1, Name: John Doe, Age: 35
ID: 2, Name: Jane Doe, Age: 30

Remaining Users:
ID: 1, Name: John Doe, Age: 35


10. load a dataset using scikit learn preprocess data and split into training and testing sets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Preprocess the data (if needed)
# In this example, I'll use StandardScaler to standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
test_size = 0.2  # Adjust the test size according to your needs
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

# Step 4: Print the shapes of the resulting sets
print("Training set - Features shape:", X_train.shape)
print("Testing set - Features shape:", X_test.shape)
print("Training set - Labels shape:", y_train.shape)
print("Testing set - Labels shape:", y_test.shape)
Training set - Features shape: (120, 4)
Testing set - Features shape: (30, 4)
Training set - Labels shape: (120,)
Testing set - Labels shape: (30,)


11.build and train a simple linear regression model using scikit learn and evaluate its performance
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Plot the regression line
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
