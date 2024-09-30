# DS-4002-Project-1
Rotten Tomatoes Movie Review text data analysis

Group Members: Erin Moulton, Hank Dickerson, Varun Pavuloori <br>

Work completed as a part of the Data Science Project course @ UVA <br>
Dataset sourced from: `https://www.kaggle.com/datasets/priyamchoksi/rotten-tomato-movie-reviews-1-44m-rows`
# Section 1
software used: Google Colab

Add-on Packages needed: pandas, matplotlib.pyplot, html, re, seaborn, nltk, nltk.sentiment.vader import SentimentIntensityAnalyzer, sklearn

Platforms: Windows, Mac

# Section 2


# Section 3
In order to reproduce the results of our study first open the link at the top of this file and download the dataset of Rotten Tomatoes movie reviews.

Using google colab read/load in the csv file of the dataset.

Drop the unnecessary columns from the dataset ('reviewId', 'creationDate', 'reviewState', 'reviewText', 'scoreSentiment').

Drop the null values and duplicates of the same review from the 'reviewText' column.

Isolate the year the movie was made from the 'creationDate' column.

From that create a new column titled 'year_created' and drop the original 'creationDate' column.

Replace HTML entities in a text string with their corresponding characters in the 'reviewText' column.

In the 'reviewText' column replace multiple spaces with a single space.

Drop all movie reviews with a creation year before 2004.

Rename columns ('id': 'movieTitle', 'reviewText': 'review', 'reviewState': 'fresh_or_rotten').

Install nltk vaderSentiment.

From nltk.sentiment.vader import SentimentIntensityAnalyzer.

Set your analyzer to SentimentIntensityAnalyzer().

Define a function with review as the sole input.

Calculate the sentiment score using the polarity scores method then compound each polarity score.

Set the sentiment values so anything inbetween -.15 and .15 will be neutral with all other negative numbers being negative and all other positive numbers being positive.

Have the function return the sentiment and the sentiment scores

Apply the function to the entire dataframe.

Use the apply method to turn the results into a new dataframe.

Obtain the average and median sentiment score for all movies labeled positive in the 'scoreSentiment' column of the original dataframe.

Obtain the average and median sentiment score for all movies labeled negative in the 'scoreSentiment' column of the original dataframe.

Import TfidfVectorizer from from sklearn.feature_extraction.text

Separate the movies using the sentiment that we just obtained.

Define a function to get the top 10 words accross all positive reviews, negative reviews, and neutral reviews.

Set the TfidfVectorizer to ignore all english stop words.

Create a TF-IDF matrix of the reviews.

Calculate the average TF-IDF score for each word in all of the reviews.

Create a dictionary putting the words with their respective scores in descending order.

Return the top words found, in this case the top 10 words.

Apply this function to all positive reviews, negative reviews, and neutral reviews seperately.

Print the top 10 words in the 3 different categories.

Next exclude top words in the positive and negative reviews that are also top words in the neutral reviews.

Install transformers, and import pipeline from transformers.

Get a random sample of 10000 movies using random_state equal to 42.

create a sentiment_pipeline variable equal to pipeline("sentiment-analysis").

Define a function that takes in a review and uses the sentiment_pipeline method to obtian a sentiment label.

Return the label of either Positive or Negative.

Apply the function to each review of the sample we just obtained.

Store the results in two new columns llm_sentiment, and llm_score.

Initialize the TfidfVectorizer to ignore english stop words.

Separate the reviews into positive and negative using the llm_sentiment.

Apply the fit_transform method to the positive and negative reviews to create two TF-IDF matrices that represent the word importance for each category.

Obtain the words used in the TF-IDF calculations.

Get a summation of the TF-IDF values of all words accross the positive and negative reviews.

Convert the summations into flat arrays.

Create a dataframe including each word and its respective positive and negative scores.

Calculate the difference between the positive score and the negative score for each word.

Create a neutral threshold of 10 to filter out extremely neutral words.

Obtain the top 10 positive and negative words based on the score differential.

Display those top 10 words for each category.

From sklearn.metrics import accuracy_score.

Define a function that will convert the sentiment of Positive or Negative into binary

Apply this function to the sentiment from the original dataframe, the sentiment found using the Vader package, and the llm sentiment we most recently found.

Find how accurately the Vader sentiment is equivalent to the original sentiment.

Do the same for the llm sentiment.

display the accuracy percentages for each.

Next we train our own model.

From sklearn.model_selection import train_test_split.

From sklearn.feature_selection import chi2.

Create a variable X which contains the movie review.

Create a variable y which contains the llm sentiment.

Split the data into 80 percent training data and 20 percent test data.

Initialize a TfidfVectorizer to convert the reviews into TF-IDF features.

Fit and Transform the vectorizer on the training and the test data.

Perform a Chi Squared test to test the importance of the TF-IDF features (words) in relation to the reviews sentiment.

Create a dataframe that includes the feature along with its respective Chi-squared statistic and p-value.

Arrange the new dataframe in descending order based on the Chi-squared statistic.

Display the top 10 words.

From sklearn.linear_model import LogisticRegression.

From sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report.

Initialize a Logistic Regression model.

Train that model on the training data set.

Use the trained model to make predictions on the sentiment of the test set and set that equal to a new variable (ex. y_pred)

Calculate the accuracy, precision, recall, and F1-score of the model.
