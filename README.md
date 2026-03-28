# Toxicity Detector (Deep Learning)

This project is a deep learning model that detects toxic comments from text.
It takes a sentence as input and predicts whether it belongs to categories like insult, threat, obscene, etc.

This project was built to understand how real-world NLP pipelines work, from text preprocessing to model training and prediction.

---

## How it works

1. The input text is converted into numbers using a TextVectorization layer
2. The processed text is passed through a Bidirectional LSTM model
3. The model outputs probabilities for different toxicity labels

---

## Tech Used

* Python
* TensorFlow / Keras
* Pandas

---

## Project Structure

* `Train.py` → trains the model
* `Predict.py` → loads model and makes predictions
* `Toxicity_Detector.h5` → saved trained model
* `.gitignore` → ignores dataset and unnecessary files

---

## Dataset

The dataset used in this project is around 66 MB and is not included in this repository.

To run the project:

1. Download the dataset from the original source from kaggle from the link "https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge"
2. Place it in the project folder
3. Update the file path in the code if needed
4. Train the model using `Train.py`
5. After training on the data test the model using `Predict.py`

---

## Why this project?

This project focuses on understanding:

* How text is converted into numerical form
* How LSTM models process sequential data
* How to maintain consistency between training and prediction

---

## Future Improvements

* Improve model accuracy
* Reduce training time
* Build a simple interface for predictions


