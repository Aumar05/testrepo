import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming df_combined is already created and cleaned
# Ensure df_combined does not have missing values and all columns are numeric

def main():
    # Load df_combined (this should be the cleaned and prepared DataFrame)
    df_combined = pd.read_csv(r"C:\Users\aumar\.vscode\pyfile\Combined.csv")
    
    # Split data into features (X) and target (y)
    X = df_combined.drop('price', axis=1)
    y = df_combined['price']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    
    # Initialize Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")
    
    # Example of using the model to predict new data
    new_data = X_test.iloc[:1]  # Taking the first row of X_test as an example
    predicted_price = model.predict(new_data)
    print(f"Predicted Price: {predicted_price}")
    
if __name__ == "__main__":
    main()
