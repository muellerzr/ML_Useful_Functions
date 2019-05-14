
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/Tabular.ipynb
import torch as torch
import numpy as np
import pandas as pd
from fastai.tabular import *
from sklearn.model_selection import train_test_split

class PrepData:
  def __init__(self, dataframe, activity):
    self.dataframe = dataframe
    dataframe['Activity'] = activity
    self.lenTrain = int(len(dataframe)/100*70)
    self.lenValid = self.lenTrain + int(len(dataframe)/100*20)
    self.lenTest = self.lenValid + int(len(dataframe)/100*10)
    self.train = dataframe.iloc[:self.lenTrain]
    self.valid = dataframe.iloc[self.lenTrain:self.lenValid]
    self.test = dataframe.iloc[self.lenValid:]

class CombineData:
  def __init__(self, df1, df2):
    self.train = df1.train.append([df2.train])
    self.valid = df1.valid.append([df2.valid])
    self.test = df1.test.append([df2.test])

def calcHiddenLayer(data, alpha, numHiddenLayers):
  if numHiddenLayers == 0:
    return []
  else:
    tempData = data.train_ds
    i, o = len(tempData.x.classes), len(tempData.y.classes)
    io = i+o
    return [(len(data.train_ds)//(alpha*(io)))//numHiddenLayers]*numHiddenLayers
  
def feature_importance(learner, cat_names, cont_names, thresh:float=0):
    loss0=np.array([learner.loss_func(learner.pred_batch(batch=(x,y.to("cpu"))), y.to("cpu")) for x,y in iter(learner.data.valid_dl)]).mean()
    fi=dict()
    types=[cat_names, cont_names]
    for j, t in enumerate(types):
      for i, c in enumerate(t):
        loss=[]
        for x,y in iter(learner.data.valid_dl):
          col=x[j][:,i]    #x[0] da hier cat-vars
          idx = torch.randperm(col.nelement())
          x[j][:,i] = col.view(-1)[idx].view(col.size())
          y=y.to('cpu')
          loss.append(learner.loss_func(learner.pred_batch(batch=(x,y)), y))
        fi[c]=np.array(loss).mean()-loss0
    d = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
    df = pd.DataFrame({'cols': [l for l, v in d], 'imp': np.log1p([v for l, v in d])})
    df = df[df['imp'] > thresh]
    cat_vars = []
    cont_vars = []
    for item in list(df.cols):
      if item in cat_names:
        cat_vars.append(item)
      if item in cont_names:
        cont_vars.append(item)
    return cat_vars, cont_vars

def SplitSet(df):
  train, test = train_test_split(df, test_size=0.1)
  return train, test


def PredictTest(df, learner):
  data = learner.data.train_ds.x
  path = learner.path
  cat_names = data.cat_names
  cont_names = data.cont_names
  procs = data.procs
  testData = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
       .split_none()
       .label_from_df(cols=dep_var)
       .databunch())
  results = learner.validate(testData.train_dl)
  acc = float(results[1]) * 100
  print("Test accuracy of: " + str(acc))
  return acc

def findBestAlpha(data):
  i = 1
  for x in range(10):
    learn = tabular_learner(data, layers=calcHiddenLayer(data, i, 2), metrics=accuracy, callback_fns=SaveModelCallback)
    if x == 0:
      print('Alpha:', i)
    else:
      print('\nAlpha:', i)
    with progress_disabled_ctx(learn) as learn:
      learn.fit_one_cycle(5)
    i += 1
