import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import IsolationForest


class MajoritySample():
  def __init__(self, labelCol:str, majorityLabel, minorityLabel, params:dict):
    super().__init__()
    self.labelCol = labelCol
    self.majorityLabel = majorityLabel
    self.minorityLabel = minorityLabel
    self.params = params
    self.scores = np.empty(0)
    self.model = None


  def filterData(self,data:pd.DataFrame):
    return data.loc[ data[self.labelCol] == self.majorityLabel , : ]


  def fit(self, data:pd.DataFrame):
    self.model = IsolationForest(**self.params).fit( self.filterData(data) )
  

  def score(self, data:pd.DataFrame):
    if self.model == None:
      self.fit(self.filterData(data))
    self.scores = self.model.decision_function(self.filterData(data))
  

  def scorePlot(self, data:pd.DataFrame):
    if self.scores.shape[0] == 0:
      self.score(data)
    sns.displot(self.scores)


  def samplePrep(self, data:pd.DataFrame, interval_sizes:list=[(.01,1000.), (.2,100.), (.5,50.), (1.,20)], density_ratio:float=1.):
    if self.scores.shape == 0:
      self.score(data)

    majorityData = self.filterData(data)
    majorityData = majorityData.loc[:,majorityData.columns]

    scoreCol = 'Isolation Scores'
    majorityData[scoreCol] = self.scores
    majorityData = majorityData.sort_values(by=[scoreCol]).reset_index()
    majorityData = majorityData.loc[:, [col for col in majorityData.columns if col != scoreCol] ]

    density = density_ratio*( data[self.labelCol] == self.minorityLabel ).sum() / data.shape[0]
    intrvl = 0 
    for tpl in interval_sizes:
      if density <= tpl[0]:
        intrvl = tpl[1]
        break

    sampleSize = int(intrvl*density) + 1
  
    return majorityData, density, intrvl, sampleSize


  def sample(self, data:pd.DataFrame, seed:int=52):
    majorityData, density, intrvl, sampleSize = self.samplePrep(data)
    
    indices = np.array( majorityData.index )
    max_ = indices.max()

    np.random.seed(seed)
    seeds = np.random.choice(np.arange(0,10*max_), int(max_/intrvl)+4, replace=False)

    left, right = 0, intrvl
    sampleIndices = np.empty(0)
    seedIndx = 0
    while right < max_:
      np.random.seed( seeds[seedIndx] )
      sampleIndices = np.hstack( ( sampleIndices , np.random.choice( np.arange(left,right), sampleSize, replace=False) ) )
      seedIndx += 1
      left += intrvl
      right += intrvl

    return majorityData.loc[ majorityData.index.intersection(sampleIndices) ]


