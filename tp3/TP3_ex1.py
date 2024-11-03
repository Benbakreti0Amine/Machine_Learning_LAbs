from builtins import print
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset for analysis.
    
    Args:
        file_path (str): Path to the CSV file
        ;
    Returns:
        tuple: Features (X) and target variable (y)
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    print("Dataset Preview:")
    print(df)
    
    # Split into features and target
    X = df.drop('Scores', axis=1)  # Hours studied
    y = df['Scores']  # Test scores
    
    return X, y

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Split the data and train the linear regression model.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        tuple: Trained model and data splits (X_train, X_test, y_train, y_test)
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def plot_regression_line(X, y, model):
    """
    Plot the dataset points and regression line.
    
    Args:
        X: Features
        y: Target variable
        model: Trained linear regression model
    """
    m = model.coef_[0]
    c = model.intercept_
    
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', label='Dataset Data')
    plt.plot(X, model.predict(X), color='red', linewidth=2, 
             label=f'Model: y = {m:.2f}x + {c:.2f}')
    plt.xlabel('Hours Studied')
    plt.ylabel('Score')
    plt.title('Data and Regression Model')
    plt.legend()
    plt.show()
    
    print(f"Model Equation: y = {m:.2f}x + {c:.2f}")

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Calculate and display model performance metrics for both training and testing sets.
    
    Args:
        model: Trained model
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing target values
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_metrics = {
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'R-squared': r2_score(y_train, y_train_pred)
    }
    
    # Calculate metrics for test set
    test_metrics = {
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'MSE': mean_squared_error(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'R-squared': r2_score(y_test, y_test_pred)
    }
    
    # Display metrics
    print("\nTraining Performance:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.2f}")
        
    print("\nTesting Performance:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.2f}")
    
    return y_test_pred

def plot_predictions(X_test, y_test, y_test_pred):
    """
    Plot actual vs predicted values for the test set.
    
    Args:
        X_test: Test features
        y_test: Actual test values
        y_test_pred: Predicted test values
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_test_pred, color='red', label='Predicted')
    plt.xlabel('Hours Studied')
    plt.ylabel('Score')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    # File path
    file_path = r"C:\Users\user\OneDrive\Desktop\projects\tp2ML\tp3\student_scores.csv"
    
    # Load and prepare data
    X, y = load_and_prepare_data(file_path)
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Plot regression line
    plot_regression_line(X, y, model)
    
    # Evaluate model and get predictions
    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Plot predictions
    plot_predictions(X_test, y_test, y_test_pred)

if __name__ == "__main__":
    main()