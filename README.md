# 💬 Sentiment Analysis Reviews Web Application

### 🔍 Analyze, Visualize, and Export Review Sentiments with Ease

This is a **Flask-based Web Application** that allows users to upload review datasets (CSV/TXT) or type reviews manually to perform **Sentiment Analysis** using **NLP (VADER & Transformers)**.  
It visualizes results using **Chart.js** and provides options to **download reports** in CSV and PDF formats.  

---

## 🚀 Project Overview

This project performs **Sentiment Analysis** on textual reviews and provides:
- ✅ Upload & Analyze reviews from CSV/TXT files  
- ✅ Automatic **Positive / Negative / Neutral** classification  
- ✅ Aspect-Based Sentiment Insights (e.g., service, quality, price)  
- ✅ Visualization through **Pie, Bar & Line Charts**  
- ✅ Dual highlighting for keywords (Positive, Negative, Aspect)  
- ✅ User Authentication (Login / Logout)  
- ✅ Export analyzed data in CSV and PDF formats  
- ✅ Interactive and modern user interface using **Bootstrap**  

---

## 🧠 Tech Stack & Tools Used

| Category | Technologies / Libraries |
|-----------|---------------------------|
| **Programming Language** | Python |
| **Framework** | Flask |
| **Database** | SQLite + SQLAlchemy ORM |
| **Authentication** | Flask-Login, JWT |
| **Frontend** | HTML, CSS, Bootstrap, JavaScript |
| **Visualization** | Chart.js |
| **NLP Libraries** | NLTK (VADER), Hugging Face Transformers |
| **Data Processing** | Pandas |
| **Server** | Flask Development Server |

---

## 📂 Folder Structure
sentiment-analysis-reviews-project/
│
├── static/ # CSS, JS, and assets
├── templates/ # HTML files (Flask Jinja2 templates)
│ ├── login.html
│ ├── register.html
│ ├── dashboard.html
│ ├── analysis.html
│ └── admin_dashboard.html
│
├── uploads/ # Uploaded CSV/TXT files
├── app.py # Main Flask backend application
├── models.py # SQLAlchemy models for User, Dataset, Review
├── sentiment_utils.py # Sentiment analysis logic (NLP)
├── requirements.txt # All dependencies
└── README.md # Project documentation (this file)


---

## 🧩 Features Explanation

### 1️⃣ **Login & Authentication**
- Users can **register and log in** using their credentials.  
- Authentication handled by **Flask-Login** and JWT tokens.  

### 2️⃣ **User Dashboard**
- Displays:
  - Total Datasets Uploaded
  - Total Reviews Analyzed
  - Overall Confidence Percentage  
- Allows users to:
  - Upload CSV/TXT datasets  
  - Enter manual reviews  
  - Start analysis  
  - View recent uploads  

### 3️⃣ **Sentiment Analysis**
- Uses **NLTK’s VADER** for polarity detection.  
- Optional integration with **Hugging Face Transformers** for deeper NLP.  
- Each review is analyzed and stored in the database with:
  - Raw Text
  - Preprocessed Text
  - Sentiment (Positive / Negative / Neutral)
  - Confidence Score  

### 4️⃣ **Aspect-Based Analysis**
- Detects key aspects like *service, quality, delivery, price, staff, etc.*  
- Shows **top positive and negative words** with their frequencies.  
- Highlights aspect-related terms within reviews.  

### 5️⃣ **Insights & Visuals**
- Interactive **Chart.js** visualizations:
  - 🥧 Pie Chart → Sentiment Distribution  
  - 📊 Bar Chart → Aspect-wise Sentiments  
  - 📈 Line Chart → Sentiment Trends Over Time  

### 6️⃣ **Dual Word Highlighting**
- Positive words → Green  
- Negative words → Red  
- Aspect words → White  
This feature visually enhances review readability.

### 7️⃣ **Export Section**
- Export all analyzed results in:
  - **CSV format** (full data)
  - **PDF format** (summary + sample reviews)

---

## ⚙️ Setup Instructions (Run Locally)

Follow these steps to run the project on your system 👇  

### 🪜 Step 1: Clone Repository

git clone https://github.com/singamsetty2020/sentiment-analysis-reviews-project.git
cd sentiment-analysis-reviews-project

🪜 Step 2: Create Virtual Environment
python -m venv venv

🪜 Step 3: Activate Virtual Environment

Windows (PowerShell)

venv\Scripts\activate


Linux / Mac

source venv/bin/activate

🪜 Step 4: Install Requirements
pip install -r requirements.txt

🪜 Step 5: Run Flask App
python app.py

🪜 Step 6: Open Browser

Visit → http://127.0.0.1:5000/

You’ll see the login page, then continue to dashboard & analysis pages.

🧮 Dataset Format

Upload files in CSV or  raw TXT format.
Your CSV must contain at least one review text column.

Example:

review
The product quality is amazing and delivery was fast!
The service was poor and the staff was rude.
The price is okay but packaging could be better.

🧾 Output & Visuals
Section	Description
Dashboard	Overview of uploads, reviews count, and confidence
Aspect Analysis	Displays aspects, top positive/negative words
Insights	Pie, Bar, and Line charts
Export	Buttons for CSV and PDF downloads

📸 (Add screenshots here later — optional)

📘 Example Review Outputs
Input Review	Sentiment	Confidence
The product is really good and fast.	Positive	92.6%
The delivery was late and packaging was bad.	Negative	88.3%
The service is okay, nothing special.	Neutral	60.5%
🧠 Future Improvements

Deploy to Render / Vercel / PythonAnywhere

Add Admin Dashboard

Integrate Deep Learning-based Transformer Model for fine-tuned predictions

Add Multi-language support

🧑‍💻 Author

👩‍💻 Lakshmi Akhila Singamsetty and team 

Fourth-Year Engineering Student | Passionate about AI, NLP, and Web Development

📧 Email: akhilasingamsetty585@gmail.com

🌐 GitHub: https://github.com/singamsetty2020

🏁 Conclusion

This project is a complete end-to-end Sentiment Analysis and Visualization System designed to simplify analyzing textual reviews and understanding user sentiments at a glance.

⭐ If you like this project, please give it a star on GitHub!
Your support means a lot 💖


---
