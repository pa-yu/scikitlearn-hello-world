# scikitlearn-hello-world

Tiny “hello world” example for supervised learning with scikit-learn:  
a decision tree that classifies fruits as **apple** or **orange** based on weight and texture.

---

## Run with Docker (recommended)

If you just want to run the example without installing anything locally:

```bash
docker run --rm ghcr.io/cmu-12780/scikitlearn-hello-world:main
```

You should see predictions for a couple of example fruits printed to the terminal.


## Build and run locally

If that fails (e.g., tag not found), you can build the image locally:

```bash
git clone https://github.com/CMU-12780/scikitlearn-hello-world.git
cd scikitlearn-hello-world
docker build -t scikitlearn-hello-world .
docker run --rm scikitlearn-hello-world
```

## Run in GitHub Codespaces

Open the repo in GitHub.

Click the green Code button → Open with Codespaces → New codespace.

Wait for the dev container to build (it uses .devcontainer/devcontainer.json and the Dockerfile).

In the Codespaces terminal, run:

```bash
python main.py
```

---

## What the example does

Builds a tiny training dataset with:

Features: weight (grams), texture (smooth = 0, bumpy = 1)

Labels: 0 = apple, 1 = orange

Trains a DecisionTreeClassifier from scikit-learn.

Uses the trained model to predict the class (and probabilities) for a couple of new fruits.



