from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

FEATURES = [
    "Age", "CGPA", "Attendance", "Technical_Skill", "Communication_Skill",
    "Aptitude_Score", "Projects", "Internships", "Certifications",
    "Extracurricular", "Coding_Score", "Soft_Skill", "Resume_Score",
    "Interview_Score"
]

COUNT_FEATURES = {"Projects", "Internships", "Certifications"}

CAREER_PROFILES = {
    "Data Scientist": {
        "Technical_Skill": 85, "Coding_Score": 80, "Aptitude_Score": 80,
        "Communication_Skill": 65, "Soft_Skill": 65, "Projects": 3,
        "Internships": 1, "Certifications": 2
    },
    "ML Engineer": {
        "Technical_Skill": 88, "Coding_Score": 88, "Aptitude_Score": 82,
        "Communication_Skill": 60, "Soft_Skill": 65, "Projects": 3,
        "Internships": 1, "Certifications": 2
    },
    "Software Developer": {
        "Technical_Skill": 82, "Coding_Score": 88, "Aptitude_Score": 75,
        "Communication_Skill": 65, "Soft_Skill": 65, "Projects": 3,
        "Internships": 1, "Certifications": 1
    },
    "Data Analyst": {
        "Technical_Skill": 72, "Coding_Score": 65, "Aptitude_Score": 78,
        "Communication_Skill": 75, "Soft_Skill": 72, "Projects": 2,
        "Internships": 1, "Certifications": 1
    },
    "Business Analyst": {
        "Technical_Skill": 60, "Coding_Score": 50, "Aptitude_Score": 82,
        "Communication_Skill": 85, "Soft_Skill": 82, "Projects": 2,
        "Internships": 1, "Certifications": 1
    },
    "AI/ML Research": {
        "Technical_Skill": 90, "Coding_Score": 85, "Aptitude_Score": 88,
        "Communication_Skill": 60, "Soft_Skill": 65, "Projects": 4,
        "Internships": 1, "Certifications": 2
    }
}

CAREER_ACTIONS = {
    "Data Scientist": ["Python + pandas", "SQL", "Statistics", "Machine learning", "3 portfolio projects"],
    "ML Engineer": ["Python", "Algorithms", "ML pipelines", "Model deployment", "2+ ML projects"],
    "Software Developer": ["DSA", "Programming language", "Git/GitHub", "Backend/API basics", "3 coding projects"],
    "Data Analyst": ["SQL", "Excel/Sheets", "Power BI/Tableau", "Statistics", "Dashboard project"],
    "Business Analyst": ["Communication", "Requirement analysis", "Excel", "SQL basics", "Case-study practice"],
    "AI/ML Research": ["Advanced mathematics", "Python", "Deep learning", "Research reading", "4 research-oriented projects"]
}

def generate_demo_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "Age": rng.integers(20, 25, n),
        "CGPA": np.round(rng.uniform(5.5, 10.0, n), 2),
        "Attendance": np.round(rng.uniform(55, 100, n), 1),
        "Technical_Skill": rng.integers(35, 101, n),
        "Communication_Skill": rng.integers(35, 101, n),
        "Aptitude_Score": rng.integers(35, 101, n),
        "Projects": rng.integers(0, 7, n),
        "Internships": rng.integers(0, 4, n),
        "Certifications": rng.integers(0, 8, n),
        "Extracurricular": rng.integers(0, 7, n),
        "Coding_Score": rng.integers(30, 101, n),
        "Soft_Skill": rng.integers(35, 101, n),
        "Resume_Score": rng.integers(35, 101, n),
        "Interview_Score": rng.integers(30, 101, n),
    })

    placement_score = (
        0.28 * data["CGPA"] * 10
        + 0.12 * data["Technical_Skill"]
        + 0.10 * data["Communication_Skill"]
        + 0.08 * data["Aptitude_Score"]
        + 0.08 * data["Coding_Score"]
        + 0.08 * data["Soft_Skill"]
        + 0.07 * data["Resume_Score"]
        + 0.07 * data["Interview_Score"]
        + 2 * data["Projects"]
        + 3 * data["Internships"]
        + data["Certifications"]
        + 0.7 * data["Extracurricular"]
        + 0.08 * data["Attendance"]
        + rng.normal(0, 8, n)
    )
    threshold = np.median(placement_score)
    data["Placement"] = (placement_score >= threshold).astype(int)
    return data

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                early_stopping=True,
                random_state=42
            ))
        ])
    }

def evaluate_models(data):
    X = data[FEATURES]
    y = data["Placement"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    trained = {}
    metrics = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = pred.astype(float)

        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
            "precision": round(precision_score(y_test, pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, pred, zero_division=0) * 100, 2),
            "roc_auc": round(roc_auc_score(y_test, proba) * 100, 2),
        }
        trained[name] = model

    return trained, metrics

def readiness_score(student):
    score = (
        0.20 * student["CGPA"] * 10
        + 0.10 * student["Attendance"]
        + 0.12 * student["Technical_Skill"]
        + 0.10 * student["Communication_Skill"]
        + 0.10 * student["Aptitude_Score"]
        + 0.10 * student["Coding_Score"]
        + 0.08 * student["Soft_Skill"]
        + 0.06 * min(student["Projects"] * 20, 100)
        + 0.05 * min(student["Internships"] * 50, 100)
        + 0.04 * min(student["Certifications"] * 20, 100)
        + 0.05 * student["Resume_Score"]
    )
    return float(max(0, min(100, score)))

def readiness_label(score):
    if score >= 80:
        return "Excellent placement readiness"
    if score >= 65:
        return "Good placement readiness"
    if score >= 50:
        return "Moderate placement readiness"
    return "Low placement readiness"

def career_matches(student):
    rows = []
    for career, requirements in CAREER_PROFILES.items():
        similarities = []
        for skill, required in requirements.items():
            actual = float(student[skill])
            if skill in COUNT_FEATURES:
                actual = min((actual / max(required, 1)) * 100, 100)
            similarities.append(max(0, 100 - abs(actual - required)))
        rows.append({"Career": career, "Match Score": round(np.mean(similarities), 2)})
    return pd.DataFrame(rows).sort_values("Match Score", ascending=False).reset_index(drop=True)

def skill_gaps(student, career):
    gaps = []
    for skill, required in CAREER_PROFILES[career].items():
        actual = float(student[skill])
        if skill in COUNT_FEATURES:
            if actual < required:
                gaps.append({
                    "Skill": skill, "Current": actual, "Required": required,
                    "Gap": round(required - actual, 2)
                })
        else:
            gap = required - actual
            if gap > 0:
                gaps.append({
                    "Skill": skill, "Current": actual, "Required": required,
                    "Gap": round(gap, 2)
                })
    return pd.DataFrame(gaps).sort_values("Gap", ascending=False) if gaps else pd.DataFrame(
        columns=["Skill", "Current", "Required", "Gap"]
    )

def feature_importance(model):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "coef_"):
        vals = np.abs(model.named_steps["model"].coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    out = pd.DataFrame({"Feature": FEATURES, "Importance": vals})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)

def predict_student(model, student):
    row = pd.DataFrame([{f: student[f] for f in FEATURES}])
    pred = int(model.predict(row)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1] * 100)
    else:
        probability = float(pred * 100)
    return pred, probability
