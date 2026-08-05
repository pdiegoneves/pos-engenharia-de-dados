import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, classification_report
    import matplotlib.pyplot as plt

    return LinearRegression, pd, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regressão Linear

    Relação entre uma variável X e uma variável dependente Y

    Variáveis com correlação forte
    """)
    return


@app.cell
def _(pd):
    # Sample dataset (house prices based on square footage)
    data = {
        'SquareFootage': [3702, 2797, 4245, 3830, 1365, 1363, 4877, 4831, 2435, 4359,
                           1205, 3962, 4531, 3050, 4149, 4869, 1990, 1521, 2235, 2946,
                           1548, 3126, 1011, 3616, 3741, 2120, 2918, 2707, 2493, 4337,
                           4590, 2745, 3635, 2468, 3980, 2097, 4453, 2973, 3044, 4228,
                           3401, 3790, 3655, 2944, 4324, 4146, 4875, 3589, 3781, 4641],
        'Price': [376261, 281803, 415105, 392345, 123417, 148268, 470262, 497420, 247031, 445138,
                  112132, 398160, 455166, 293451, 416123, 489381, 213318, 153822, 222256, 297898,
                  155716, 314165, 123710, 350507, 374737, 199845, 304127, 278622, 256162, 428960,
                  470167, 266494, 365957, 249820, 408388, 215719, 440702, 303099, 303755, 418299,
                  338325, 390252, 376964, 294970, 434157, 404675, 494030, 354036, 376958, 454888]
    }

    df = pd.DataFrame(data)
    df
    return (df,)


@app.cell
def _(df, train_test_split):
    X = df[['SquareFootage']] # Features
    Y = df[['Price']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Training data: {X_train.shape}, {Y_train.shape}")
    print(f"Testing data: {X_test.shape}, {Y_test.shape}")
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(LinearRegression, X_train, Y_train):
    model = LinearRegression()

    model.fit(X_train, Y_train)

    print(f"Intercept: {model.intercept_}")
    print(f"Coeficiente: {model.coef_[0]}")
    return (model,)


@app.cell
def _(X_test, Y_test, model):
    y_pred = model.predict(X_test)

    print("Predicted Prices:", y_pred)
    print("Actual Prices:", Y_test)
    return (y_pred,)


@app.cell
def _(X_test, Y_test, plt, y_pred):
    plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')

    plt.xlabel('Square Footage')
    plt.ylabel('Price')
    plt.title('House Price vs Square Footage')
    plt.legend()

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
