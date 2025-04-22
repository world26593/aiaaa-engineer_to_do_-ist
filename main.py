import streamlit as st
import json
import os
from datetime import datetime, timedelta

DATA_FILE = "ai_engineer_progress.json"

# Estimated hours to complete each task (for realistic tracking)
estimated_hours = 3  # You can customize this per task later

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

# Load or initialize progress
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        progress = json.load(f)
else:
    progress = {phase: {task: False for task in tasks} for phase, tasks in default_tasks.items()}

st.set_page_config(page_title="AI Engineer Roadmap", layout="wide")
st.title("🧠 AI Engineer Roadmap Tracker")

# Calculate overall progress
total_tasks = sum(len(tasks) for tasks in progress.values())
completed_tasks = sum(1 for phase in progress.values() for done in phase.values() if done)
progress_percent = int((completed_tasks / total_tasks) * 100)
remaining_hours = (total_tasks - completed_tasks) * estimated_hours
expected_days = remaining_hours // 2  # If user studies ~2 hrs/day

st.markdown("## 📊 Overall Progress")
st.progress(progress_percent / 100)
st.success(f"{completed_tasks} / {total_tasks} tasks completed ({progress_percent}%)")
st.info(f"⏳ Remaining Time Estimate: {remaining_hours} hours (~{expected_days} days)")
st.caption(f"📅 If you study ~2h/day, you can finish by {datetime.now() + timedelta(days=expected_days):%d %B %Y}")

# Show phase-wise progress
for phase, tasks in progress.items():
    st.markdown(f"---\n### 📌 {phase}")
    phase_total = len(tasks)
    phase_done = sum(1 for done in tasks.values() if done)
    phase_percent = int((phase_done / phase_total) * 100)
    phase_remaining = (phase_total - phase_done) * estimated_hours
    phase_days = phase_remaining // 2
    phase_end_date = datetime.now() + timedelta(days=phase_days)

    col1, col2 = st.columns([4, 2])
    with col1:
        st.progress(phase_percent / 100)
    with col2:
        st.caption(f"{phase_done}/{phase_total} tasks ({phase_percent}%)")
        st.caption(f"🕐 Est. time left: {phase_remaining}h (~{phase_days} days)")
        st.caption(f"📆 Est. Finish: {phase_end_date:%d %b %Y}")

    for task in tasks:
        checked = st.checkbox(f"✅ {task}", value=progress[phase][task], key=phase + task)
        progress[phase][task] = checked

# Save progress
with open(DATA_FILE, "w") as f:
    json.dump(progress, f)
