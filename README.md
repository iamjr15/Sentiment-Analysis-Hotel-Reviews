
# Hotel Review Sentiment Analysis 

This project performs semantic analysis on hotel reviews scraped from TripAdvisor to classify them as positive, negative or neutral. Both traditional machine learning and deep learning approaches are implemented for comparison.

## Data Overview

The dataset contains over 20,000 real TripAdvisor reviews for hotels with the following attributes:

- Review Text - The actual review content
- Rating - Score from 1 to 5 given by the reviewer 

## Data Import and Storage

The CSV dataset is loaded into a pandas dataframe for ease of manipulation and analysis using vectorized operations. This also allows seamless data visualization through matplotlib and seaborn. For bigger datasets, leveraging numpy arrays or SQL databases would be more efficient.

## Exploratory Data Analysis 

Statistical analysis on the raw data reveals insights like:

- Rating distribution is skewed towards positive reviews
- Higher rated hotels have longer review lengths on average 
- Word frequency analysis finds phrases associated with sentiment

Such findings guide the data preprocessing and modeling steps.

## Data Preprocessing

The major data cleaning and transformation tasks applied are:

**Text Normalization** - Lowercasing, punctuation removal, stopwords filtering
**Lemmatization** - Word conversion to base dictionary form  
**Rating Simplification** - Binning ratings to 3 categories 

This retains the core semantic content while easing modeling.

## Machine Learning Modeling

A TfidfVectorizer converts the preprocessed text into TF-IDF vectors retaining key phrases and their significance. Multiple classification models are trained on this vectorized data:

- Logistic Regression 
- Random Forest
- SVM 
- Naive Bayes
- KNN

Logistic regression emerged most accurate through 10-fold stratified cross-validation. The model is further tuned through an exhaustive grid search over hyperparameters like regularization strength and solver algorithms.

## Deep Learning Modeling

The neural network architecture consists of:

- Embedding Layer - Learns latent vector representations for words
- LSTM Layer - Captures semantic context and long-range dependencies
- Dense Layers - Perform high-level reasoning and classification

The model is trained for 5 epochs with Adam optimization and categorical cross-entropy loss. Validation accuracy reaches 85% reflecting robust learned embeddings.

## Prediction Testing

Both models are able to effectively classify complex reviews conveying subtle sentiment based on the demonstration. Their probabilistic confidence scores also provide a calibrated measure of certainty.
