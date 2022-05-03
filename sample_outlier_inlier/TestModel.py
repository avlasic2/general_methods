import pandas as pd

from MajoritySample import MajoritySample

string = 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download'
data = pd.read_csv('creditcard.csv')

params = {
  'n_estimators':100
, 'max_samples':'auto'
, 'contamination':'auto'
, 'max_features':1.0
, 'bootstrap':False
, 'n_jobs':None
, 'random_state':256
}
labelCol = 'Class'
majorityLabel = 0
minorityLabel = 1

sampler = MajoritySample(labelCol=labelCol, majorityLabel=majorityLabel, minorityLabel=minorityLabel, params=params)
sampler.scorePlot(data) #this method displays the histogram of the scores, however, one may also use the 'score' method if not interested in visualization
sampled_data = sampler.sample(data)