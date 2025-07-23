# SPAM-MESSAGE-DETECTION-TASK-4

*COMPANY*: CODTECH IT SOLUTIONS 

*NAME*:SANJAI KRISHNAN S 

*INTERN ID*:CTO4DG3176

*DOMAIN*: PYTHON PROGRAMMING 

*DURATION*:4 WEEKS

*MENTOR*:NEELA SANTOSH 

*PROCESS DESCRIPTION*:

The Spam Message Detection using NLP and ML project demonstrates a practical application of Natural Language Processing (NLP) and Machine Learning (ML) to solve a real-world problem—identifying spam messages. Developed by Sanjai Krishnan. S as part of Task 3 during an internship at Codetech IT Solutions, this project classifies text messages as either "spam" or "ham" (non-spam) using a supervised learning approach with TF-IDF vectorization and the Naive Bayes classifier.


---

Step-by-Step Process:

1. Understanding the Objective

The core aim of the project is to build a simple and interpretable machine learning model that can distinguish between legitimate and spam messages. The solution focuses on the textual content of messages and uses classic NLP techniques to convert human-readable text into machine-readable numerical features.

2. Setting Up the Environment

Essential Python libraries were imported at the start:

Pandas for data manipulation.

Matplotlib and Seaborn for data visualization.

Scikit-learn for model training, feature extraction, and evaluation.


These libraries provide a robust toolkit for building end-to-end ML pipelines.

3. Creating and Preprocessing the Dataset

Instead of using an external file, a small sample dataset containing 10 messages labeled as "ham" or "spam" was manually created using a Python dictionary and converted into a DataFrame. Labels were then encoded into binary values (ham as 0 and spam as 1) for compatibility with the machine learning algorithm.

4. Data Splitting

The dataset was split into training and testing subsets using train_test_split() from Scikit-learn, with 70% of the data used for training and 30% for testing. This ensures that the model is evaluated on unseen data, providing a fair assessment of its predictive capabilities.

5. Feature Extraction with TF-IDF

Since machine learning algorithms cannot work directly with raw text, the messages were converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This technique measures how important a word is to a message in the context of the entire dataset. The TfidfVectorizer was used to both transform the training data and apply the same transformation to the test data.

6. Model Training with Naive Bayes

The Multinomial Naive Bayes algorithm was selected for its effectiveness in text classification problems. It is particularly suitable for datasets with discrete features like word counts or TF-IDF scores. The model was trained on the vectorized training data and learned to associate word patterns with spam or ham labels.

7. Model Evaluation

After training, the model was evaluated on the test data. Performance metrics included:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)


A heatmap of the confusion matrix was plotted using Seaborn to visualize the model's predictions vs. actual labels.

8. Custom Message Prediction

To demonstrate real-world application, a few custom messages were manually tested. These were vectorized using the trained TF-IDF vectorizer and classified using the trained Naive Bayes model. The predicted output was printed as either "Spam" or "Ham" based on the model's judgment.

Conclusion:

This project provides a strong foundation for understanding how NLP and machine learning techniques can be applied to filter unwanted or harmful messages. While the dataset is small for demonstration purposes, the process mimics real-world spam detection systems used in email services and messaging platforms. Future improvements could include larger datasets, advanced NLP techniques (like stemming or lemmatization), and deep learning models. This hands-on task significantly enhanced the intern’s understanding of data preprocessing, feature engineering, and text classification workflows.

*OUTPUT*:

![Image](https://github.com/user-attachments/assets/cffd00f6-a92b-4495-b96c-9807655e8af9)
