# 🧭 Ideal Types of Travelers — Customer Segmentation Tool

This project provides a framework for analyzing travel-related data to identify different customer segments based on behavior, preferences, and sentiment.

## 🚀 Project Objectives

- Cluster travelers based on booking behavior, demographics, and survey responses
- Perform sentiment analysis on travel review comments
- Generate actionable insights to support marketing and product strategies

## 📊 Features

- K-Means and DBSCAN clustering models to segment customer groups
- Sentiment analysis pipeline for processing user feedback
- Modular and scalable design — easy to adapt to your own datasets

## 🔄 How to Use

1. Prepare your own datasets:
   - Booking details or customer profiles (CSV format)
   - Review comments or feedback (text-based)
   - Survey responses or additional metrics

2. Adjust file paths inside `main.py` and `segment.py` to point to your files.

3. Run the main script:

```bash
python main.py

The system will:

Clean and preprocess the data

Perform clustering and sentiment analysis

Generate an output CSV with labeled customer segments

🧾 File Overview
File	Description
main.py	Coordinates the entire data processing pipeline
segment.py	Contains clustering models and logic
sentiment_analysis.py	Processes text comments using sentiment scoring
clustered_customers.csv	Output with customers segmented into groups (auto-generated)

💼 Use Cases
Identify “price-sensitive”, “experience-driven”, or “luxury” customer types

Align product offerings with travel behavior trends

Use insights to improve customer experience and marketing ROI

📦 Technologies Used
Python 3

Pandas

scikit-learn

seaborn / matplotlib

TextBlob / NLTK for sentiment analysis

👤 Author
Maintained by Kyle Walkerley
