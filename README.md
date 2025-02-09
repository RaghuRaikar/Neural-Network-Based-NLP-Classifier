🧠 Neural Network-Based NLP Classifier
======================================

🚀 **Text Classification & Named Entity Recognition with RNNs, LSTMs, and Viterbi Algorithm**

This project implements **neural network-based text classification** and a **sequence labeling model using the Viterbi algorithm**. It trains models for **sentiment analysis** and **named entity recognition (NER)** using deep learning techniques such as **Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs), and the Viterbi algorithm**.

* * * * *

🎯 **Project Overview**
-----------------------

This NLP classifier is designed to:\
✅ **Perform Sentiment Analysis** 📊 (Positive/Negative classification)\
✅ **Recognize Named Entities (NER)** 🔎 (Person, Location, Organization)\
✅ **Compare RNNs & LSTMs for text classification**\
✅ **Implement the Viterbi Algorithm for structured sequence prediction**

📌 The models are trained and tested on **IMDb movie reviews** (for sentiment analysis) and the **CoNLL 2003 NER dataset** (for named entity recognition).

* * * * *

🛠 **Implemented Techniques**
-----------------------------

🔹 **Text Classification** (Sentiment Analysis)

-   Uses RNNs and LSTMs
-   Compares accuracy of both models
-   Evaluates on IMDb movie reviews dataset

🔹 **Named Entity Recognition (NER)**

-   Uses the **Viterbi Algorithm** for sequence labeling
-   Implements a structured prediction model

🔹 **Optimization Techniques**

-   Word embedding
-   Dropout regularization
-   Adam optimizer

* * * * *

📥 **Dataset & Inputs**
-----------------------

The project uses two primary datasets:

📌 **IMDb Reviews Dataset** 🎭

-   Used for **sentiment classification**
-   Classifies reviews as **positive or negative**

📌 **CoNLL 2003 Named Entity Recognition Dataset** 🏷

-   Used for **NER**
-   Labels entities as **Person (PER), Location (LOC), Organization (ORG), or Other (O)**

### 🔢 **Input Format**

-   Text data for classification (movie reviews or entity-tagged sentences)
-   Word embeddings for deep learning models

### 📤 **Output Format**

-   **For Sentiment Analysis:** Predicted sentiment (Positive/Negative)
-   **For NER:** Tagged words with entity labels

* * * * *

📊 **Results & Performance**
----------------------------

### **Sentiment Analysis Results**

| Model | Training Accuracy | Test Accuracy |
| --- | --- | --- |
| RNN | 87.57% | 65.09% |
| LSTM | 99.93% | 84.08% |

👉 **LSTMs outperform RNNs** significantly, particularly on longer sequences.

### **NER Results (Viterbi Algorithm)**

| Dataset | Precision | Recall | F1 Score |
| --- | --- | --- | --- |
| Dev | 52% | 45% | 48% |
| Test | 54% | 46% | 47% |

👉 The **Viterbi Algorithm** achieves **48% F1-score** on the dev set, **47% on test**.

* * * * *

🚀 **How to Run the Project**
-----------------------------

### 📥 1. **Clone the Repository**

`git clone https://github.com/your-repo/NLP-Classifier.git`  
`cd NLP-Classifier`

### 🏗 2. **Install Dependencies**

`pip install -r requirements.txt`

### ▶️ 3. **Train & Evaluate the Models**

#### **Train Sentiment Analysis Models**

`python main.py --task sentiment --model rnn  # Train RNN model`  
`python main.py --task sentiment --model lstm  # Train LSTM model`

#### **Train Named Entity Recognition (NER) Model**

`python main.py --task ner`

#### **Evaluate with Viterbi Algorithm**

`python conlleval.py ner.test output.txt`

* * * * *

📦 **Project Structure**
------------------------

- 📂 **NLP-Classifier**  
- ├── 📄 `text_classification_rnn_lstm.py` *(Main script to train & evaluate text classification models using RNNs & LSTMs)*  
- ├── 📄 `viterbi_sequence_labeling.py` *(Implementation of the Viterbi algorithm for sequence labeling tasks, such as Named Entity Recognition (NER))*  
- ├── 📄 `conlleval.py` *(NER evaluation script for computing precision, recall, and F1-score based on CoNLL format output)*  
- ├── 📄 `model.simple` *(Pre-trained NER model used for inference and evaluation)*  
- ├── 📄 `ner.train` *(Training dataset for Named Entity Recognition (NER), containing labeled sequences of text)*  
- ├── 📄 `ner.dev` *(Development dataset for fine-tuning and hyperparameter selection in NER models)*  
- ├── 📄 `ner.test` *(Test dataset for final performance evaluation of the trained NER model)*  
- ├── 📄 `README.md` *(Comprehensive documentation of the project! 🚀)*  

* * * * *

📌 **Key Takeaways**
--------------------

✅ **LSTMs outperform RNNs** for text classification\
✅ **Viterbi Algorithm is crucial** for structured sequence prediction\
✅ **Deep learning can extract sentiment & named entities** effectively

* * * * *

🔥 **Future Enhancements**
--------------------------

🔹 **Use Transformer-based models (BERT, GPT)** for improved accuracy\
🔹 **Enhance Named Entity Recognition with CRF layer**\
🔹 **Improve hyperparameter tuning for better results**

* * * * *

📩 **Contributing**
-------------------

Have suggestions? Found a bug? Open an issue or submit a pull request! 🚀

* * * * *

🏆 **Credits & References**
---------------------------

-   📚 **CoNLL 2003 Dataset** for Named Entity Recognition
-   🎭 **IMDb Dataset** for Sentiment Analysis
-   🔗 TensorFlow & Keras documentation

✨ **A powerful NLP classifier using deep learning! 🚀**
