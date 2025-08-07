
# 📄 Resume Analysis and Skills-Based Job Recommendation System

A machine learning-powered system that analyzes resumes using NLP techniques and recommends relevant jobs along with personalized skill improvement suggestions. Built with Python, Flask, and SQLite, this system streamlines recruitment by matching candidate profiles with suitable job descriptions.

---

## 📌 Problem Statement

Recruiters face challenges manually reviewing thousands of resumes, leading to time inefficiency and suboptimal hiring. Online job portals lack real-time personalization and matching precision. This system automates:

- Resume parsing
- Skill matching
- Personalized job recommendation
- Suggestions for courses to improve employability

---

## 🎯 Objectives

- ✅ Analyze candidate resumes using NLP
- ✅ Extract and compare job descriptions
- ✅ Recommend relevant jobs based on matching score
- ✅ Provide skill-gap suggestions to improve resume quality

---

## 💡 Proposed Solution

1. **Data Extraction**  
   - Extract text from resumes & job descriptions using NLP (spaCy)

2. **Text Preprocessing**  
   - Tokenization, stop word removal, stemming, lemmatization

3. **Feature Matching**  
   - Use TF-IDF to vectorize text  
   - Apply Cosine Similarity to match candidate and job

4. **Ranking & Filtering**  
   - Use KNN to find the closest job matches  
   - Suggest missing skills & certifications

5. **Personalization**  
   - Adjust recommendations based on preferences like industry, location, and roles

---

## 🧪 Technologies Used

| Component          | Technology                  |
|--------------------|------------------------------|
| 💻 Frontend        | Flask                        |
| 🧠 Backend         | Python                       |
| 🗃️ Database        | SQLite                       |
| 🔬 Libraries       | spaCy, Scikit-learn, Pandas, NumPy, TF-IDF |
| 🛠 Tools           | VS Code, Jupyter Notebook    |
| 💽 OS Requirement  | Windows / Linux              |

---

## ⚙️ Algorithms

- **TF-IDF**: Converts resumes & job descriptions into numerical vectors  
- **Cosine Similarity**: Measures how similar two text documents are  
- **KNN (K-Nearest Neighbors)**: Recommends closest job postings based on vector similarity  

---

## 📊 System Architecture

```plaintext
[ Resume Upload ] --> [ NLP Extraction ] --> [ Preprocessing ] --> [ TF-IDF + Cosine Similarity ] --> [ KNN Recommendations ] --> [ Output: Job & Skill Suggestions ]

🧩 Features
🔍 Resume keyword extraction (skills, education, experience)

🧠 Intelligent job matching using similarity scores

📈 Skill-gap analysis and upskilling suggestions

💬 Real-time, personalized job recommendations

📂 Stores shortlisted/not selected resumes in Excel

