import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# ---------------- UI STYLE ----------------
st.set_page_config(page_title="Auto ML Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.glass {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Auto Model Selector")


# ---------------- AUTO MODEL CLASS ----------------
class AdvancedAutoModelSelector:

    def __init__(self, problem_type="classification"):
        self.problem_type = problem_type
        self.models = self._init_models()

    def _init_models(self):

        if self.problem_type == "classification":
            return {
                "Logistic Regression": (
                    LogisticRegression(max_iter=1000),
                    {"model__C": [0.1, 1, 10]}
                ),
                "Random Forest": (
                    RandomForestClassifier(),
                    {"model__n_estimators": [50, 100]}
                ),
                "SVM": (
                    SVC(),
                    {"model__C": [0.1, 1]}
                )
            }

        else:
            return {
                "Linear Regression": (
                    LinearRegression(),
                    {}
                ),
                "Random Forest": (
                    RandomForestRegressor(),
                    {"model__n_estimators": [50, 100]}
                ),
                "SVR": (
                    SVR(),
                    {"model__C": [0.1, 1]}
                )
            }

    def train(self, X, y, progress_bar):

        results = {}
        trained_models = {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        total = len(self.models)
        step = 0

        for name, (model, params) in self.models.items():

            step += 1
            progress_bar.progress(step / total)

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            grid = GridSearchCV(pipe, params, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            preds = best_model.predict(X_test)

            if self.problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)

            results[name] = score
            trained_models[name] = best_model

            time.sleep(0.5)

        best_model_name = max(results, key=results.get)

        return best_model_name, results, trained_models


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)

    problem_type = st.selectbox(
        "🧠 Select Problem Type",
        ["classification", "regression"]
    )

    if st.button("🚀 Train Models"):
        X = df.drop(columns=[target])
        y = df[target]

        selector = AdvancedAutoModelSelector(problem_type)

        progress = st.progress(0)

        best_model, results, models = selector.train(X, y, progress)

        # -------- Results Table --------
        st.success(f"🏆 Best Model: {best_model}")

        res_df = pd.DataFrame(results.items(), columns=["Model", "Score"])
        st.dataframe(res_df)

        # -------- Plot 1: All Models --------
        st.subheader("📊 Model Comparison")

        fig1, ax1 = plt.subplots()
        ax1.bar(results.keys(), results.values())
        ax1.set_ylabel("Score")
        ax1.set_title("Model Scores")
        st.pyplot(fig1)

        # -------- Plot 2: Best Model --------
        st.subheader("⭐ Best Model Score")

        fig2, ax2 = plt.subplots()
        ax2.bar([best_model], [results[best_model]])
        ax2.set_ylabel("Score")
        ax2.set_title("Best Model Performance")
        st.pyplot(fig2)
