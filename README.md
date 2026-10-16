# CareerCompass AI – Student Success & Placement Intelligence System

A complete VS Code-ready Streamlit application based on the uploaded CareerCompass AI
documentation and notebook.

## What this model does

CareerCompass AI integrates:

1. Student profile and academic/placement attributes
2. Placement prediction
3. Placement probability and readiness score
4. Six-model ML comparison:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - SVM
   - KNN
   - Neural Network
5. Top-3 personalized career recommendations
6. Skill-gap analysis
7. Personalized improvement roadmap
8. Feature-importance view
9. CSV upload for testing a student
10. Downloadable student report
11. Downloadable prediction CSV
12. Demo dataset generation
13. Clear distinction between demo/synthetic data and real institutional data

## Important data note

The supplied notebook uses a synthetic/demo dataset of 1,200 students. Its placement target is
also generated from a project-defined formula plus random variation. Therefore, the prediction
shown by this application is a demonstration of the proposed system and must NOT be presented
as a validated real-world placement probability until trained and evaluated on a genuine
institutional placement dataset.

## Project structure

CareerCompass_AI_VSCode/
│
├── app.py
├── ml_engine.py
├── report_generator.py
├── train_models.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── careercompass_demo.csv
│
├── models/
│   └── (generated automatically)
│
└── reports/
    └── (generated/downloaded reports)

## Run in VS Code

### 1. Open the folder

Open `CareerCompass_AI_VSCode` in VS Code.

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

The browser will open automatically. If it does not, open the local URL printed by Streamlit.

## Optional: retrain models

```bash
python train_models.py
```

The app also trains automatically if model artifacts are missing.

## Expected workflow

Student Profile
→ Data Validation
→ ML Prediction
→ Placement Probability
→ Placement Readiness
→ Career Matching
→ Skill-Gap Analysis
→ Personalized Roadmap
→ Download Report

## CSV format

For batch prediction, a CSV should contain the following columns:

Age, CGPA, Attendance, Technical_Skill, Communication_Skill,
Aptitude_Score, Projects, Internships, Certifications,
Extracurricular, Coding_Score, Soft_Skill, Resume_Score,
Interview_Score

A `Placement` column is optional. If present, the app can also compare predictions with the
provided target.

## Team

CareerCompass AI – Student Success & Placement Intelligence System

Team:
- Utkalika Priyadarshinee Champati – Team Lead
- Viswitha – Team Member
- Bhimavarapu Venkata Naga Divyasri – Team Member
