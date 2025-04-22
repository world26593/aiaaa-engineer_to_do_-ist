import streamlit as st
import json
import os

DATA_FILE = "ai_engineer_progress.json"

default_tasks = {
    "Phase 0: Programming & Math": [
        "Python (Core + Libraries like NumPy, Pandas, Matplotlib)",
        "C++ or Java (optional but good for system-level skills)",
        "Git & GitHub",
        "Math: Linear Algebra",
        "Math: Calculus",
        "Math: Probability & Statistics"
    ],
    "Phase 1: Machine Learning": [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Decision Trees",
        "Random Forests",
        "Naive Bayes",
        "SVM",
        "Gradient Boosting",
        "XGBoost, CatBoost, LightGBM"
    ],
    "Phase 2: ML Evaluation & Optimization": [
        "Train-Test Split",
        "K-Fold Cross Validation",
        "Confusion Matrix, Precision, Recall, F1",
        "Hyperparameter Tuning: Grid/Random Search",
        "Model Selection and Bias-Variance Tradeoff"
    ],
    "Phase 3: Deep Learning": [
        "Neural Networks Basics",
        "Backpropagation",
        "CNN",
        "RNN",
        "LSTM",
        "Transformers",
        "Autoencoders",
        "GANs"
    ],
    "Phase 4: Frameworks": [
        "Scikit-learn",
        "PyTorch",
        "TensorFlow",
        "Keras (optional)"
    ],
    "Phase 5: NLP & CV": [
        "Text preprocessing",
        "TF-IDF, Word2Vec",
        "BERT & Transformers",
        "OpenCV",
        "Image classification with CNN",
        "Object Detection"
    ],
    "Phase 6: MLOps": [
        "Model Deployment (Flask/FastAPI)",
        "Docker",
        "MLflow / DVC",
        "CI/CD",
        "Streamlit Dashboard",
        "Model Drift Monitoring"
    ],
    "Phase 7: Projects & Portfolio": [
        "Build ML Projects",
        "Build DL Projects",
        "Portfolio Website",
        "Upload Projects to GitHub",
        "Blog Technical Write-ups"
    ],
    "Phase 8: Advanced Topics": [
        "Reinforcement Learning",
        "Federated Learning",
        "TinyML",
        "Edge AI"
    ]
}

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        progress = json.load(f)
else:
    progress = {phase: {task: False for task in tasks} for phase, tasks in default_tasks.items()}

st.title("AI Engineer Roadmap Tracker")

total_tasks = sum(len(tasks) for tasks in progress.values())
completed_tasks = sum(1 for phase in progress.values() for done in phase.values() if done)
progress_percent = int((completed_tasks / total_tasks) * 100)
st.progress(progress_percent / 100)
st.write(f"Overall Progress: {completed_tasks} / {total_tasks} tasks completed ({progress_percent}%)")

for phase, tasks in progress.items():
    phase_total = len(tasks)
    phase_done = sum(1 for done in tasks.values() if done)
    phase_percent = int((phase_done / phase_total) * 100)
    st.subheader(f"{phase} ({phase_done}/{phase_total} - {phase_percent}%)")
    for task in tasks:
        checked = st.checkbox(task, value=progress[phase][task], key=phase + task)
        progress[phase][task] = checked

with open(DATA_FILE, "w") as f:
    json.dump(progress, f)
