# scikitlearn-hello-world

Tiny “hello world” example in main.py for supervised learning with scikit-learn:  
a decision tree that classifies fruits as **apple** or **orange** based on weight and texture.

The file concrete_strenth.py uses pandas and scikit-learn to build a regression model that predicts concrete compressive strength based on mixture ingredients (cement, water, slag, fly ash, aggregates, etc.).

The dataset comes from the UCI Machine Learning Repository and is included in this repository under concrete/.

The project includes:

- Data loading and exploration (head, describe, correlations, shape)
- Train/test split with train_test_split
- Feature scaling with StandardScaler
- Model training using RandomForestRegressor
- Performance evaluation using MSE, RMSE, R²
- Feature importance analysis

---

## Run with Docker (recommended)

If you just want to run the example without installing anything locally:

```bash
docker run --rm ghcr.io/cmu-12780/scikitlearn-hello-world:concrete_strength
```

You should see predictions for a couple of example fruits printed to the terminal.


## Build and run locally

If that fails (e.g., tag not found), you can build the image locally:

```bash
git clone https://github.com/pa-yu/scikitlearn-hello-world.git
cd scikitlearn-hello-world
pip install -r requirements.txt
python concrete_strength.py
```

## Run in GitHub Codespaces

Open the repo in GitHub.

Click the green Code button → Open with Codespaces → New codespace.

Wait for the dev container to build (it uses .devcontainer/devcontainer.json and the Dockerfile).

In the Codespaces terminal, run:

```bash
python concrete_strength.py
```

---

## What the example does

Builds a tiny training dataset with:

Features: weight (grams), texture (smooth = 0, bumpy = 1)

Labels: 0 = apple, 1 = orange

Trains a DecisionTreeClassifier from scikit-learn.

Uses the trained model to predict the class (and probabilities) for a couple of new fruits.



