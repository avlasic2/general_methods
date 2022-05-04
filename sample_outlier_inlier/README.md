# Using Isolation Forests to Sample the Majority Class
This technique utilizes [isolation forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) to identify outlier and inlier data points in a set through the scoring of the model. The technique then sorts the records with the scores and samples uniformly form an interval to ensure outlier data points are utilized in the training set and test set. 

For the example the data set from the [Kaggle credit card fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) challange is applied to the class to display how to utilize the technique. 
