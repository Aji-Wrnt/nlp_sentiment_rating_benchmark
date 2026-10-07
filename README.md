# Mobile Game Review Rating Prediction Pipeline Using NLP, Machine Learning, and Deep Learning

## Project Overview

This project implements an end-to-end Natural Language Processing (NLP) and predictive modeling pipeline to predict user rating scores (ranging from 1 to 5) based on textual review content extracted from the Google Play Store. The target case study focuses on English-language user reviews for the game Mobile Legends: Bang Bang (com.mobile.legends).

The workflow covers all stages of a machine learning lifecycle: data scraping, exploratory data analysis (EDA), text preprocessing, feature engineering through text representations (TF-IDF and Word2Vec), multi-class classification using traditional Machine Learning (Logistic Regression, Random Forest) and Deep Learning (LSTM), comparative performance evaluation, and handling severe class imbalance using SMOTE.

---

## Dataset

- Data Source: Google Play Store user reviews for Mobile Legends: Bang Bang (package: com.mobile.legends)
- Total Samples: 2,000 reviews
- Language: English (lang='en', country='us')
- Attributes:
  - review: Textual comment submitted by the user
  - rating: Numerical score assigned by the user (scale 1 to 5)
- Distribution Characteristics: The dataset displays prominent class imbalance, with ratings 1 and 5 comprising the vast majority of samples, while intermediate ratings (2, 3, and 4) constitute a minor fraction.

---

## Pipeline Stages

### 1. Data Collection (Scraping)
User reviews were harvested programmatically using the google-play-scraper library, retrieving 2,000 recent reviews sorted by newest submissions. The extracted dataset is stored in review_game.csv.

### 2. Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to inspect the structure and linguistic characteristics of the dataset:
- Evaluation of rating frequency distributions.
- Identification of dominant and top-frequency terms across the entire corpus and segmented per rating class.
- Analysis of informal language, typos, and elongated slang words characterized by repeating characters.

### 3. Text Preprocessing
Raw review texts undergo a cleaning and normalization sequence to reduce noise:
- Case Folding: Conversion of all text to lowercase.
- Character Cleaning: Stripping numbers, punctuation marks, and non-alphanumeric symbols using regular expressions.
- Tokenization: Splitting text into individual token units via NLTK word tokenization.
- Stopword Removal: Eliminating standard English stopwords and filtering out short tokens with lengths under three characters.
- Lemmatization: Transforming tokens into their dictionary base forms (lemmas) using WordNet Lemmatizer to preserve grammatical categories and semantic meaning.

### 4. Text Representation
Cleaned texts are transformed into numerical vector spaces via two distinct feature extraction methods:
1. TF-IDF (Term Frequency - Inverse Document Frequency):
   - Computes statistical weighting reflecting word importance within individual reviews relative to the entire corpus. High-frequency words across all documents receive lower weights, while domain-informative terms receive higher weights.
2. Word2Vec:
   - Encodes context-dependent continuous word vectors trained using a shallow artificial neural network.
   - For Machine Learning classifiers, document-level vectors are computed using average vector pooling (mean pooling).
   - For Deep Learning, reviews are formatted as sequential matrices with fixed-length padding and truncating.

### 5. Machine Learning Modeling and Hyperparameter Tuning
Two supervised classification algorithms were implemented and trained across both TF-IDF and Word2Vec representations:
1. Logistic Regression:
   - A linear probabilistic classification model.
   - Hyperparameter tuning via GridSearchCV (3-fold cross validation): regularization strength parameter C and maximum iterations max_iter.
2. Random Forest Classifier:
   - An ensemble bagging model based on randomized decision trees.
   - Hyperparameter tuning via GridSearchCV (3-fold cross validation): number of estimators n_estimators and maximum tree depth max_depth.

A total of four Machine Learning configurations were trained and evaluated.

### 6. Deep Learning Modeling and Hyperparameter Tuning
To capture temporal dependencies and sequential semantic patterns within review texts, a recurrent architecture was constructed:
- Architecture: Long Short-Term Memory (LSTM) receiving Word2Vec continuous sequence inputs.
- Layer Topology: Sequential LSTM layer, Dropout layer for regularization, Dense hidden layer with ReLU activation, and a final 5-unit Dense layer with Softmax activation for multi-class classification.
- Hyperparameter Tuning: Grid comparison across unit counts, dropout probabilities, and batch sizes.
- Training Optimization: EarlyStopping callback monitoring validation loss to prevent overfitting.

### 7. Performance Evaluation
Evaluation was conducted on a held-out test split (20 percent of data) using stratified sampling. Evaluated performance metrics include:
- Accuracy
- Precision (Weighted Average)
- Recall (Weighted Average)
- F1-Score (Weighted Average)

Results are synthesized into structured comparative summary tables and multi-metric bar chart visualisations comparing representation methods and algorithmic performance.

### 8. Handling Imbalanced Data
To investigate whether correcting class disparities improves classification performance on underrepresented classes, a data sampling experiment was carried out:
- Sampling Method: Synthetic Minority Over-sampling Technique (SMOTE) was applied exclusively to the training set to prevent data leakage into test evaluations.
- Model Retraining: All Machine Learning and Deep Learning architectures were retrained on the resampled training distribution.
- Comparative Impact Analysis: A side-by-side performance comparison (pre-balancing versus post-balancing) analyzing the behavior of sparse TF-IDF spaces, dense Word2Vec document vectors, and sequential LSTM embeddings under synthetic interpolation.

---

## Directory Structure

```text
.
|-- review_game.csv   # Scraped dataset containing 2,000 raw reviews and ratings
|-- train.ipynb       # Jupyter Notebook containing the complete implementation code and analysis
|-- README.md         # Comprehensive project documentation
```

---

## Environment and Dependencies

This project requires Python 3.x and the following libraries:
- pandas
- numpy
- scikit-learn
- nltk
- gensim
- tensorflow
- imbalanced-learn
- matplotlib
- google-play-scraper

---

## Instructions for Execution

1. Ensure all required Python dependencies listed above are installed in your working environment.
2. Open `train.ipynb` using Jupyter Notebook, JupyterLab, or Visual Studio Code.
3. Execute the notebook cells sequentially from top to bottom:
   - Section 1: Data scraping and generation of review_game.csv.
   - Section 2: Exploratory data analysis and visualization.
   - Section 3: Text preprocessing and generation of clean_review.csv.
   - Section 4: Dataset splitting and feature extraction (TF-IDF and Word2Vec).
   - Section 5: Machine Learning training and hyperparameter tuning.
   - Section 6: Deep Learning model implementation, tuning, and training.
   - Section 7: Unified performance evaluation and metric visualization.
   - Section 8: SMOTE rebalancing, retraining, and comparative analysis.
