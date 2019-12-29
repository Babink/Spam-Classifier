import pandas as pd

dataset = pd.read_table(
        filepath_or_buffer="./SMSSpamCollection",
        sep="\t",
        header=None,
        names=["labels" , "message"]
        )

dataset.head()


# let's label spam = 1 and ham = 0 , so that it will be easy for library to train our model
dataset["labels"] = dataset.labels.map({'ham':0 , 'spam': 1})
print(dataset.shape)
dataset.head()

# need to convert message column into integer as well , best solution will be to use bag of words method
# for testing purpose let's use bag of words in our dummy data

documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']


# this will create a vector of every word as a row

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(documents)
count_vector.get_feature_names()

# let's convert our vector into a matrix
# matrix with row being documnets items and columns being bad of words

doc_array = count_vector.transform(documents).toarray()

# let's now convert this array into a dataframes
frequency_matrix = pd.DataFrame(doc_array , columns=count_vector.get_feature_names())

# now let's train our model
# first let's split our dataset into training and testset
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(dataset["message"] , dataset["labels"] , random_state=1)


# applying bag of words 
# this will create vector of every word as a row
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)


# let's use naive bayes theorem to predict 
# why naive_bayes, cause it is suitable for classification with discrete features (word counts)
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data , y_train)

# now let's make predication on our naive_bayes model
predication = naive_bayes.predict(testing_data)

# let's evaluate our model 
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
print("Accuracy score: {}".format(accuracy_score(y_test , predication)))
print("Precision Score: {}".format(precision_score(y_test , predication)))
print("Recall Score: {}".format(recall_score(y_test , predication)))
print("F1 Score: {}".format(f1_score(y_test , predication)))