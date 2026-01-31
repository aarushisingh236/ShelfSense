# 📚 ShelfSense  
### A Personal Reading Decision System

ShelfSense is a data-driven web application designed to help users manage their personal book library and decide what to read next based on preferences, reading habits, and current mood.

Instead of relying on complex machine learning pipelines, ShelfSense focuses on **explainable recommendation logic** using semantic similarity and structured scoring, making its suggestions transparent and easy to understand.

---

## ✨ Features

- 📖 **Personal Library Management**
  - Add, edit, delete, and view books
  - Store metadata such as genre, rating, priority, status, and series information

- 📊 **Analytics Dashboard**
  - Visualize reading patterns by status, priority, rating category, and book length
  - Gain insights into reading behavior

- 🔍 **Search & Filtering**
  - Search books by title, author, and genre
  - Filter recommendations using user-defined criteria

- 🤖 **Recommendation System**
  - Content-based recommendations using TF-IDF and cosine similarity
  - Filter-based shortlisting (rating, priority, format, type)

- 📌 **What Should I Read Next?**
  - Suggests the next book based on:
    - A recently liked book
    - Genre similarity
    - Predicted enjoyment score
    - Reading status and priority

- 🧠 **Mood-Based Reading**
  - Slider-based mood inputs (Light, Comfort, Fast-Paced, Thoughtful, Intense)
  - Produces tailored recommendations based on current mindset

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI framework
- **Pandas** – data handling
- **Matplotlib** – analytics visualization
- **Scikit-learn** – TF-IDF & cosine similarity
- **CSV-based storage** (Book_List_Cleaned.csv)

---

## 📂 Project Structure

```text
ShelfSense/
│
├── app.py
├── Book_List_Cleaned.csv
├── README.md
└── requirements.txt
```
## 🚀 Getting Started
### Clone the Repository
```bash
git clone https://github.com/your-username/shelfsense.git
cd shelfsense
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Application
```bash
streamlit run app.py
```
---
## 🧪 Recommendation Logic
ShelfSense uses a hybrid scoring approach:
### Content similarity:
- TF-IDF vectorization of summary, genre, and author
- cosine similarity

### Structured scoring:
- Rating
- Priority
- Series bonus
- Genre overlap
- Mood weights
- 
---

## 💡 Future Enhancements

Planned improvements for future versions of **ShelfSense** include:

- 🔐 **User Authentication** – Allow multiple users to maintain their own personal libraries  
- 🌙 **Dark Mode UI** – Improve accessibility and visual comfort  
- 📤 **Export Recommendations** – Download reading lists as CSV or PDF  
- 📊 **Interactive Charts (Plotly)** – Replace static charts with dynamic visualizations  
- 📱 **Mobile UI Optimization** – Enhance layout for smaller screens  
- ☁️ **Google Drive Sync** – Backup and restore book data automatically  
- 📚 **Multiple Libraries Support** – Manage separate collections (e.g., Fiction, Academic, Personal)  
- 🗄 **Cloud Database Integration** – Replace CSV storage with a scalable cloud database (Firebase / PostgreSQL)

These features aim to make ShelfSense more scalable, personalized, and production-ready.
