import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regrassão logística
    É um modelo de classificação
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    return (
        LogisticRegression,
        accuracy_score,
        classification_report,
        confusion_matrix,
        np,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("study_hours_pass.csv")
    df
    return (df,)


@app.cell
def _(df, train_test_split):
    X = df[['StudyHours']]
    Y = df[['Pass']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(f"Dados de Treino: {X_train.shape}, {Y_train.shape}")
    print(f"Dados de Teste: {X_test.shape}, {Y_test.shape}")
    return X, X_test, X_train, Y_test, Y_train


@app.cell
def _(LogisticRegression, X_train, Y_train):
    model = LogisticRegression()

    model.fit(X_train, Y_train)
    return (model,)


@app.cell
def _(model):
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficient: {model.coef_[0]}")
    return


@app.cell
def _(X_test, Y_test, model):
    y_pred = model.predict(X_test)

    print("Predicted Outcomes (Pass/Fail): ", y_pred)
    print("Actual Outcomes: ", Y_test.values)
    return (y_pred,)


@app.cell
def _(Y_test, accuracy_score, classification_report, confusion_matrix, y_pred):
    accuracy = accuracy_score(Y_test, y_pred)

    conf_matrix = confusion_matrix(Y_test, y_pred)

    class_report = classification_report(Y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix")
    print(conf_matrix)
    print("Classification Report")
    print(class_report)
    return


@app.cell
def _(X, X_test, Y_test, model, np, plt):
    study_hours_range = np.linspace(X.min(), X.max(), 100)

    y_prob = model.predict_proba(study_hours_range.reshape(-1, 1))[:, 1]

    plt.scatter(X_test, Y_test, color='blue', label='Actual Data')

    plt.plot(study_hours_range, y_prob, color='red', label='Logistic Regression Curve')

    plt.xlabel('Study Hours')
    plt.ylabel('Probability of Pasing')
    plt.title('Logistic Regression: Study Hours vs. Pass')
    plt.legend()

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
