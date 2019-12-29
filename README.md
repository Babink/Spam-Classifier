# Spam Classifier
   this spam classifier uses *Naive-Bayes* Algorithms to predict accuracy of give email 
   * Accuracy Score
   * Precision Score
   * Recall Score
   * F1 Score
   
# Libraries
	* pandas
	* scikit learn
	
# Process
	* Convert Words into Bag of Words (so it will be easy for Algorithm to preprocess and learn from patterns)
	* Use Scikit learn's feature extraction method CountVectorizer to convert words into vector
	* Split dataset into training and testing sets

# WHY NAIVE BAYES?
	*navie bayes is suitable for classification with discrete features*