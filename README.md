# Sentiment Analysis using BERT

This project implements a sentiment analysis system that classifies text as **Positive** or **Negative** using Natural Language Processing (NLP) techniques. Multiple models were explored and trained, and the best-performing model was selected and deployed for real-time usage.

---

## About the Project

The goal of this project is to understand the sentiment expressed in text data such as reviews or user feedback. The project starts with traditional machine learning approaches to establish a baseline and then moves to an advanced transformer-based deep learning model for better performance.

The final system allows a user to enter any text through a simple user interface, sends that text to a backend model, and returns the predicted sentiment.

---

## Models Used

### Baseline Models

* **Logistic Regression**
* **Naive Bayes**

These models were trained using **TF-IDF vectorized text features**. They help establish a performance baseline and allow comparison with deep learning models.

### Final Model

* **BERT (bert-base-uncased)**

BERT is a transformer-based pretrained language model that understands contextual meaning in text. It was fine-tuned on the IMDB movie reviews dataset for supervised binary classification (Positive / Negative).

---

## Model Training Approach

* Text data was cleaned and preprocessed
* Labels were encoded (Positive → 1, Negative → 0)
* Data was split into training and testing sets
* Baseline ML models were trained using TF-IDF features
* BERT was fine-tuned using transfer learning
* Performance of all models was compared
* BERT was selected as the final model
* The trained BERT model was saved and reused for inference

---

## Final Output

The final trained BERT model is integrated into a Streamlit-based user interface. When a user enters text, the backend model predicts the sentiment and returns the result as:

* **Positive**
* **Negative**

---

## Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Google Colab
