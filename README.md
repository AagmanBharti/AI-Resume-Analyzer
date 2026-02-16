## 🚀 Enterprise AI Resume Analyzer

An intelligent, AI-powered Resume Analysis and ATS Evaluation system that helps candidates optimize their resumes and enables recruiters to efficiently rank applicants using Machine Learning and NLP techniques.

## 📌 Overview

The Enterprise AI Resume Analyzer evaluates resumes against job descriptions using hybrid scoring techniques that combine keyword matching, semantic similarity, and structural ATS checks.

The system supports:

👨‍💻 Candidate Resume Evaluation

🧑‍💼 Recruiter Resume Ranking

🤖 AI-based Resume Category Prediction

📊 ATS Score & Skill Gap Analysis

📄 Automated PDF Evaluation Reports

## ✨ Features
🔹 Candidate Mode

Upload resume (PDF)

Job Description matching

AI Resume Category Prediction

Skill Extraction & Missing Skill Detection

Hybrid ATS Score Calculation

Resume Section Analysis

Match Score Visualization

Downloadable Resume Evaluation Report

🔹 Recruiter Mode

Bulk Resume Upload

Automatic Candidate Ranking

CSV Export of Candidate Scores

## 🧠 AI & NLP Techniques Used

TF-IDF Keyword Matching

Semantic Similarity Scoring

Skill Extraction using NLP

Resume Classification using Machine Learning

Hybrid ATS Scoring Model

Text Preprocessing & Lemmatization

## 🛠️ Tech Stack
Programming

Python

Machine Learning & NLP

Scikit-learn

NLTK

TF-IDF Vectorization

Frontend / UI

Streamlit

Data Processing

Pandas

NumPy

Matplotlib

Utilities

Joblib (Model Serialization)

ReportLab (PDF Report Generation)

### 📂 Project Structure


RESUME/
│
├── data/
│   └── resume_dataset.csv
│
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── preprocess.py
│   ├── scoring.py
│   ├── semantic_match.py
│   ├── skills.py
│   ├── report.py
│   └── train.py
│
├── utils/
│   └── pdf_reader.py
│
├── app.py
├── requirements.txt
└── README.md


## ⚙️ Installation & Setup
1️⃣ Clone Repository - 

git clone https://github.com/YOUR_USERNAME/Resume-Analyzer.git

cd Resume-Analyzer

2️⃣ Create Virtual Environment - 

python -m venv venv


Activate environment:

Windows - 

venv\Scripts\activate

Mac/Linux - 

source venv/bin/activate

3️⃣ Install Dependencies - 

pip install -r requirements.txt

4️⃣ Run Application - 

streamlit run app.py

## 📊 ATS Scoring Methodology

The system evaluates resumes using:

✔ Job Description Match Score

✔ Skill Strength Score

✔ Resume Structure Validation

✔ Semantic Similarity Matching

✔ Resume Length Optimization

## 📄 Report Generation

The system generates downloadable PDF reports containing:

Resume Category

Match Score Breakdown

Skill Analysis

Missing Skill Suggestions

ATS Quality Score

## 📌 Example Use Cases

👨‍🎓 Students & Job Seekers

Improve resume quality

Identify missing skills

Optimize ATS compatibility

🧑‍💼 Recruiters

Automate candidate screening

Rank resumes efficiently

Reduce manual evaluation time

## 🚀 Future Improvements

Transformer-based semantic scoring

Resume content improvement suggestions using LLMs

Skill proficiency scoring

Cloud-based deployment scaling

Advanced recruiter analytics dashboard

## 🤝 Contributing

Contributions, feature suggestions, and improvements are welcome.

Steps:

Fork repository

Create feature branch

Submit Pull Request

## 📜 License

This project is open-source and available under the MIT License.

## 👤 Author

Aagman Bharti

GitHub: https://github.com/AagmanBharti

LinkedIn: https://www.linkedin.com/in/aagman-bharti-a05917288/

## ⭐ If You Found This Useful

Give the repository a star to support development!
