# Fake-News-Detection  

The Fake News Detection system classifies news articles or headlines as "Fake" or "Real". It uses a trained machine learning model that analyzes the textual content of the news to make predictions. The project demonstrates the complete pipeline—from data preprocessing and model training to evaluation and deployment.




## ✨Importing Libraries and Dataset 

## here we are using 

- **Pandas** – To load the Dataframe
- **Matplotlib** – To visualize the data features i.e. barplot
- **spaCy** – Advanced NLP tasks (optional alternative to NLTK)
-**scikit-learn** – Machine learning models (e.g., Logistic Regression, Naive Bayes), feature extraction (TF-IDF, CountVectorizer), model evaluation

 ## ✨ Key Features of Fake-news-detection.
## 🧹 Text Preprocessing:

- Cleans and tokenizes raw news text using NLP techniques (stopword removal, stemming, punctuation cleanin


## 📚 Feature Extraction:

- Converts text into numerical representations using methods like TF-IDF and Count Vectorizer for model training.


## 🧠 Multiple ML Models:

- Logistic Regression

- Naive Bayes

- Random Forest& XGBoost


## 📊 Model Evaluation:

- Evaluates model performance using accuracy, precision, recall, and F1-score, along with confusion matrix visualizatio.


  ## **🚀 Getting Started**
 ##  📁 Load the Dataset:
Import a labeled dataset containing real and fake news articles.

**here is the code**:-

**import pandas as pd
df = pd.read_csv('fake_news.csv')**


## 🧹 Data Preprocessing:
- Removing punctuation, numbers, and special characters.
- Converting to lowercase.
- Applying stemming/lemmatization

**here is the code**:-

**from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer**


## 📊 Evaluate the Model:-


**from sklearn.metrics import accuracy_score, confusion_matrix, classification_report**.


## Visualization of Results:

## 🧾 Confusion Matrix:-

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()**


## 📈 Classification Report (Text Output):-


from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))


These visualizations help confirm that the model not only performs well numerically but also behaves as expected when applied to real-world data....








