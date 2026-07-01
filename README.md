# Movie Review Sentiment Analyzer

This is a simple web app that predicts whether a movie review is positive or negative. It uses a Naive Bayes model trained on the IMDB dataset and is built with Streamlit.

## How to run it

- Clone the repo to your computer
  ```
  git clone <your repo link here>
  ```

- Go into the project folder
  ```
  cd <folder name>
  ```

- Install the required libraries
  ```
  pip install -r requirements.txt
  ```

- Download the dataset from Kaggle (IMDB Dataset.csv) and put it in the same folder

- Run the training script to generate the model file (only need to do this once)
  ```
  python train_model.py
  ```

- Run the app
  ```
  streamlit run app.py
  ```

- It should open automatically in your browser at http://localhost:8501
