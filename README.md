
# 📲 SMS Spam Detection System 🚫✉️  
> **Automatically classify SMS messages as spam or not spam using machine learning!**  

## 📝 Project Overview  
The SMS Spam Detection project aims to automatically classify text messages as spam or ham (not spam) using various machine learning techniques. It involves preprocessing SMS data, transforming the text into numerical features, and training multiple models to predict whether a message is spam. This project can be used as a foundation for applications like spam filtering in emails and messaging services. 🛡️📧  

### 🔧 Key Components  
- **Data Preprocessing:** Cleaning, tokenizing, and normalizing SMS text data.  
- **Feature Extraction:** Using TF-IDF vectorization to convert text into numerical representations.  
- **Model Training:** Comparing different machine learning models to find the best-performing one.  
- **Performance Evaluation:** Using metrics like accuracy, precision for evaluation.  

---

## 🎥 Demo  
Here's an example of how the model works:  
| **SMS Message**                                     | **Prediction** |  
|----------------------------------------------------|----------------|  
| "Congratulations! You've won a $1000 gift card!"   | **Spam**       |  
| "Hey, can we reschedule our meeting to tomorrow?"  | **Ham**        |  

---

## ✨ Features  
- **Real-time SMS Spam Detection:** Predict whether an incoming message is spam or ham.  
- **Multiple Model Comparison:** Evaluate various machine learning algorithms for classification.  
- **Visualization Tools:** Graphs and plots for better understanding of model performance.  
- **Pretrained Models:** Use saved models for real-time predictions on new data.  
- **Text Preprocessing Techniques:** Convert raw text into meaningful features for model training.  

---

## 📊 Dataset  
The dataset used is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), which consists of approximately 5,572 SMS messages labeled as "spam" or "ham." Each message is preprocessed to remove punctuation, convert to lowercase, and handle stopwords.

| **Column**   | **Description**                      |  
|--------------|--------------------------------------|  
| `Label`      | Indicates if the message is spam or ham. |  
| `Message`    | The actual text of the SMS message.        |  

---

## 🧰 Technologies Used  
### Languages and Libraries  
- **Python 3.6+**: The programming language used.  
- **Libraries:**  
  - **scikit-learn**: For machine learning algorithms and model evaluation.  
  - **pandas**: Data manipulation and analysis.  
  - **numpy**: For numerical operations.  
  - **matplotlib & seaborn**: Data visualization and plotting.  
  - **nltk (Natural Language Toolkit)**: Text preprocessing and tokenization.  

---

## 📝 Usage  
1. **Open the Jupyter Notebook:**  
   ```bash  
   jupyter notebook sms-spam-detection.ipynb  
   ```  
2. **Follow the steps in the notebook:**  
   - Load the dataset and explore it.  
   - Preprocess the text data.  
   - Train and evaluate different classification models.  
   - Save the trained model and vectorizer for future predictions.  
3. **Using the pretrained model:**  
   - Load the `model.pkl` and `vectorizer.pkl` files to make predictions on new SMS messages.  

---

## 📈 Model Performance  
The project compares several models to identify the best-performing algorithm for spam detection:  

| **Model**                     | **Accuracy** | **Precision** | **Recall** | **F1-Score** |  
|-------------------------------|--------------|---------------|------------|--------------|  
| **Naive Bayes**               | 97.09%       | 100%          | 95.0%      | 97.44%       |  
| **Random Forest**             | 97.58%       | 98.29%        | 96.5%      | 97.38%       |  
| **Support Vector Classifier** | 97.58%       | 97.48%        | 97.5%      | 97.48%       |  
| **Logistic Regression**       | 95.84%       | 97.03%        | 92.5%      | 94.72%       |  

### Visualization  
The notebook includes visualizations to help interpret the model's performance, such as confusion matrices and ROC curves. 📊🔍  

---

## 🌱 Future Improvements  
- **Advanced Preprocessing:** Incorporate stemming, lemmatization, or word embeddings (e.g., Word2Vec, GloVe).  
- **Hyperparameter Tuning:** Use grid search or random search for optimizing model parameters.  
- **Model Deployment:** Create a web interface using Flask or deploy the model as an API.  
- **Real-time Data Integration:** Connect with a messaging service to classify incoming SMS messages.  

---
