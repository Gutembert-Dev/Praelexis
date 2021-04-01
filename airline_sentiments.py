# coding: utf-8


import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify import NaiveBayesClassifier

tweets = pd.read_csv('c:/Users/Gutembert Nzeugaing/Desktop/GUT/PY/DA/Tweets.csv')

# Divide the tweets into three groups
has_positive_sentiment = tweets['airline_sentiment'].str.contains('positive')
has_negative_sentiment = tweets['airline_sentiment'].str.contains('negative')
has_neutral_sentiment  = tweets['airline_sentiment'].str.contains('neutral')

positive_tweets = tweets[has_positive_sentiment]
# Pre-train tests
positive_shape = positive_tweets.shape
# print(positive_shape)
negative_tweets = tweets[has_negative_sentiment]
# Pre-train tests
negative_shape = negative_tweets.shape
# print(negative_shape)
neutral_tweets = tweets[has_neutral_sentiment]
# Pre-train tests
neutral_shape = neutral_tweets.shape
# print(neutral_shape)

# Find the best and the worst Airline in the month of February 2015
# The best Airline:
best_airline = positive_tweets[['airline', 'airline_sentiment_confidence']]
count_best_airline = best_airline.groupby('airline', as_index=False).count()
count_best_airline = count_best_airline.sort_values('airline_sentiment_confidence', ascending=False)
# print(count_best_airline)
# The worst airline
worst_airline = negative_tweets[['airline', 'airline_sentiment_confidence', 'negativereason']]
count_worst_airline = worst_airline.groupby('airline', as_index=False).count()
count_worst_airline = count_worst_airline.sort_values('negativereason', ascending=False)
# print(count_worst_airline)
# Find the common problems caused a bad flight. (For this, consider only the negative tweets)
reason = negative_tweets[['airline', 'negativereason']]
count_bad_flight_reason = reason.groupby('negativereason', as_index=False).count()
count_bad_flight_reason = count_bad_flight_reason.sort_values('negativereason', ascending=True)
# print(count_bad_flight_reason)

# Model Training:

def build_bag_of_words_features_filtered(words):
    bag_of_words_features_filtered = {}
    for word in words:
        if word in bag_of_words_features_filtered:
            bag_of_words_features_filtered[word] = bag_of_words_features_filtered[word] + 1
        else:
            bag_of_words_features_filtered[word] = 1
    return bag_of_words_features_filtered

# Train to classify the tweets into positive and negative

# Tweets into positive features
positive_features = []
for text in positive_tweets['text']:
    positive_features.append(((build_bag_of_words_features_filtered(word_tokenize(text)), 'pos')))
# print(positive_features)

# Tweets into negative features
negative_features = []
for text in negative_tweets['text']:
    negative_features.append(((build_bag_of_words_features_filtered(word_tokenize(text)), 'neg')))
# print(negative_features)

# Tweets into neutral features
neutral_features = []
for text in neutral_tweets['text']:
    neutral_features.append(((build_bag_of_words_features_filtered(word_tokenize(text)), 'neu')))
# print(neutral_features)

# print(len(positive_features))
split = 2000
sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])
# print(sentiment_classifier)
sentiment_accuracy = nltk.classify.util.accuracy(sentiment_classifier, positive_features[:split]+negative_features[:split])*100
# print(sentiment_accuracy)
positive_features_verify = positive_features[split:]
# print(positive_features_verify)
negative_features_verify = negative_features[split:2363]
# print(negative_features_verify)
sentiment_accuracy_verify = nltk.classify.util.accuracy(sentiment_classifier, positive_features_verify+negative_features_verify)*100
# print(sentiment_accuracy_verify)

# Train to classify the negative tweets on the reason
has_costumer_service_issue = negative_tweets['negativereason'].str.contains("Customer Service Issue")
costumer_service_issue = negative_tweets[has_costumer_service_issue]
# Pre-train tests
costumer_service_issue_shape = costumer_service_issue.shape
# print(costumer_service_issue_shape)
costumer_service_issue_features = []
for text in costumer_service_issue['text']:
    costumer_service_issue_features.append(((build_bag_of_words_features_filtered(word_tokenize(text)), 'neservice_issue')))
# print(costumer_service_issue_features)

has_late_flight = negative_tweets['negativereason'].str.contains("Late Flight")
late_flight = negative_tweets[has_late_flight]
# Pre-train tests
late_flight_shape = late_flight.shape
# print(late_flight_shape)
late_flight_features = []
for text in late_flight['text']:
    late_flight_features.append(((build_bag_of_words_features_filtered(word_tokenize(text)), 'late_flight')))
# print(late_flight_features)

test = has_costumer_service_issue | has_late_flight 
others = negative_tweets[~test]
# Pre-train tests
others_shape = others.shape
# print(others_shape)
other_features = []
for text in others['text']:
    other_features.append(((build_bag_of_words_features_filtered(word_tokenize(text)), 'other')))
# print(other_features)

# print(len(late_flight_features))
split = 1000
bad_cause_classifier = NaiveBayesClassifier.train(costumer_service_issue_features[:split]+late_flight_features[:split]+other_features[:split])
reason_accuracy = nltk.classify.util.accuracy(bad_cause_classifier, costumer_service_issue_features[:split]+late_flight_features[:split]+other_features[:split])*100
# print(reason_accuracy)
costumer_service_verify = costumer_service_issue_features[split:1400]
late_flight_verify = late_flight_features[split:1400]
others_verify = other_features[split:1400]
reason_accuracy_verify = nltk.classify.util.accuracy(bad_cause_classifier, costumer_service_verify+late_flight_verify+others_verify)*100
# print(reason_accuracy_verify)