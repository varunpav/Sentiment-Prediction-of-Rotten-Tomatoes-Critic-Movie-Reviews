# DS-4002-Project-1
Rotten Tomatoes Movie Review text data analysis

Group Members: Erin Moulton, Hank Dickerson, Varun Pavuloori <br>

Work completed as a part of the Data Science Project course @ UVA <br>
Dataset sourced from: `https://www.kaggle.com/datasets/priyamchoksi/rotten-tomato-movie-reviews-1-44m-rows`
# Section 1
software used: Google Colab

Add-on Packages needed: pandas, matplotlib.pyplot, html, re, seaborn, nltk, nltk.sentiment.vader import SentimentIntensityAnalyzer, sklearn

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

