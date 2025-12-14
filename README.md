# Sentiment Analysis using BERT

This project implements a sentiment analysis system that classifies text as **Positive** or **Negative** using Natural Language Processing (NLP) techniques. Multiple models were explored and trained, and the best-performing model was selected and deployed for real-time usage.

---

## About the Project

I built a sentiment analysis system using the IMDB movie reviews dataset to classify text as positive or negative. I first trained baseline machine learning models like Logistic Regression and Naive Bayes using TF-IDF features to establish a benchmark. After that, I fine-tuned a pretrained BERT transformer model using transfer learning, which captured contextual meaning in text more effectively and achieved better performance. The trained BERT model was saved and integrated into a Streamlit-based user interface, where user input is sent to the backend model and the predicted sentiment is returned in real time.

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

<img width="1493" height="564" alt="Screenshot 2025-12-14 141147" src="https://github.com/user-attachments/assets/e6f50fb2-9f28-493a-ab6c-afdc19cf6ef2" />

<img width="452" height="393" alt="download" src="https://github.com/user-attachments/assets/a3a59783-28c4-4e01-b5a3-c0fa3070a5a3" />

<img width="613" height="374" alt="download (1)" src="https://github.com/user-attachments/assets/dffb40f1-8e87-4cd2-84e4-8e2cc5c48c8a" />

<img width="846" height="451" alt="download (2)" src="https://github.com/user-attachments/assets/7e00edd8-265f-4e49-9994-0d057e18d977" />

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
