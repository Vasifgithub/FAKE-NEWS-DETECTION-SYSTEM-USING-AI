
# 🕵️♂️ Fake News Detection System Using AI

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A powerful machine learning web application that detects fake news articles.
This project is a web-based Fake News Detection system that uses a pre-trained XGBoost model to classify whether a given news text is Real or Fake (Rumor).
Built using Flask, Bootstrap 5, and XGBoost, this app provides a simple and attractive user interface where users can paste a news article and get instant predictions.

![Demo Screenshot](image.png)

## Features
-  Text input interface for news articles
-  Real-time predictions using XGBoost model
-  Modern UI with gradient design and animations
-  Color-coded results (Green for Real/Red for Fake)
-  Input validation and error handling
-  Fully responsive design

## 🛠️ Technologies Used
**ML & NLP**  
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-blue)
![TF-IDF](https://img.shields.io/badge/TF--IDF-Vectorization-yellowgreen)

**Backend**  
![Flask](https://img.shields.io/badge/Flask-2.3.2-lightgrey)
![Gunicorn](https://img.shields.io/badge/Gunicorn-21.2.0-important)

**Frontend**  
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.0-purple)
![CSS3](https://img.shields.io/badge/CSS3-Animation-blue)

**Deployment**  
![Render](https://img.shields.io/badge/Deployment-Render-blueviolet)


## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Git LFS (for model files)

```bash
# Clone repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Install Git LFS and pull models
git lfs install
git lfs fetch --all
git lfs pull

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

```
Passionate about AI, Deep Learning, and building meaningful projects.
