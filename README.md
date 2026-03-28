#Toxicity Detector 

This project is a simple deep learning model that detects toxic comments from text. It takes a sentence as input and predicts whether it belongs to categories like insult, threat, obscene, etc.

I built this project to understand how real-world NLP pipelines work — from text preprocessing to model training and prediction.

---

How it works

The input text is converted into numbers using a TextVectorization layer 2. The processed text is passed through a Bidirectional LSTM model 3.

---

Tech Used

* Python * TensorFlow / Keras * Pandas, NumPy

---

Project Structure

py` → trains the model * `predict. py` → loads model and makes predictions * `Toxicity_Detector. h5` → saved trained model * `vocab. txt` → saved vocabulary for consistent predictions * `.

---

Dataset

The dataset used in this project is around 66 MB and is not included in this repository.

To run the project:

Download the dataset from the original source 2. Place it in the project folder 3.

---

Important Note

The model depends on the same vocabulary used during training. txt` must be present while making predictions.

---

Why this project?

This project focuses more on understanding:

* How text is converted into numerical form * How LSTM models process sequential data * How to maintain consistency between training and prediction

---

Future Improvements

* Improve model accuracy * Reduce training time * Build a simple web interface for predictions

---

Author

Made as part of a college AI/ML project.
