import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("concrete/Concrete_Data.xls", engine="xlrd")

def load_dataset():
    print("First 5 Rows")
    print(df.head())
    print()

    print("Data Shape:")
    print(df.shape)
    print()

    print("Summary Statistics:")
    print(df.describe())
    print()

    print("Correlation to target:")
    corr = (
        df.corr(numeric_only=True)["Concrete compressive strength(MPa, megapascals) "]
        .sort_values(ascending=False)
    )
    print(corr)
    print()

    # seperating data variables
    y = df["Concrete compressive strength(MPa, megapascals) "].to_numpy()
    X = df.drop(columns=["Concrete compressive strength(MPa, megapascals) "]).to_numpy()
    
    return X, y

def main():
    X, y = load_dataset()

    print("Training data shape:", X.shape)
    print("Target shape:", y.shape)
    print()

    # splitting into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test)

    # train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # making predictions
    predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Performance:")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"RMSE: {np.sqrt(mse):.3f}")
    print(f"R² Score: {r2:.3f}")
    print()
    
    # showing feature importances
    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": df.columns[:-1], 
        "importance": importances
    })

    df_imp_sorted = df_imp.sort_values("importance", ascending=False)

    print("Feature importances:")
    print(df_imp_sorted, "\n")

if __name__ == "__main__":
    main()

