
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/Tabular.ipynb
import torch as torch
import numpy as np
import pandas as pd
import os
import subprocess
from fastai.tabular import *
from fastai.utils.mod_display import *
from fastai.callbacks import *
from sklearn.model_selection import train_test_split

def plotFunctions():
  os.chdir('..')
  os.rename('content/ML_Useful_Functions/basic_train.py', 'usr/local/lib/python3.6/dist-packages/fastai/basic_train.py')
  os.chdir('content')
  os._exit(00)
  return

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

def calcHiddenLayer(data, alpha, numHiddenLayers:int = 2):
  if numHiddenLayers == 0:
    return []
  else:
    tempData = data.train_ds
    i, o = len(tempData.x.classes), len(tempData.y.classes)
    io = i+o
    return [(len(data.train_ds)//(alpha*(io)))//numHiddenLayers]*numHiddenLayers
  
def feature_importance(learner, top_n:int = 5, return_table:bool = False): 
  # based on: https://medium.com/@mp.music93/neural-networks-feature-importance-with-fastai-5c393cf65815
    data = learner.data.train_ds.x
    cat_names = data.cat_names
    cont_names = data.cont_names
    loss0=np.array([learner.loss_func(learner.pred_batch(batch=(x,y.to("cpu"))), y.to("cpu")) for x,y in iter(learner.data.valid_dl)]).mean()
    #The above gives us our ground truth for our validation set
    fi=dict()
    types=[cat_names, cont_names]
    for j, t in enumerate(types): # for all of cat_names and cont_names
      for i, c in enumerate(t):
        loss=[]
        for x,y in iter(learner.data.valid_dl): # for all values in validation set
          col=x[j][:,i] # select one column of tensors
          idx = torch.randperm(col.nelement()) # generate a random tensor
          x[j][:,i] = col.view(-1)[idx].view(col.size()) # replace the old tensor with a new one
          y=y.to('cpu')
          loss.append(learner.loss_func(learner.pred_batch(batch=(x,y)), y))
        fi[c]=np.array(loss).mean()-loss0
    d = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
    df = pd.DataFrame({'cols': [l for l, v in d], 'imp': np.log1p([v for l, v in d])})
    cat_vars, cont_vars = [],[]
    for x in range(top_n):
      if df['cols'].iloc[x] in cat_names:
        cat_vars.append(df['cols'].iloc[x])
      if df['cols'].iloc[x] in cont_names:
        cont_vars.append(df['cols'].iloc[x])
    if return_table:
      return cat_vars, cont_vars, pd.DataFrame({'cols': [l for l, v in d], 'imp': np.log1p([v for l, v in d])})
    else:
      return cat_vars, cont_vars

def SplitSet(df):
  train, test = train_test_split(df, test_size=0.1)
  return train, test


def PredictTest(df, learn, dep_var):
  data = learn.data.train_ds.x
  path = learn.path
  cat_names = data.cat_names
  cont_names = data.cont_names
  procs = data.procs
  testData = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
       .split_none()
       .label_from_df(cols=dep_var)
       .databunch())
  results = learn.validate(testData.train_dl)
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
    
    
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='accuracy', mode:str='auto', every:str='improvement', name:str='bestmodel',):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name = every,name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

                 
    def jump_to_epoch(self, epoch:int)->None:
        try: 
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                #print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save(f'{self.name}')

    def on_train_end(self, **kwargs):
        "Load the best model."
        if self.every=="improvement" and (self.learn.path/f'{self.learn.model_dir}/{self.name}.pth').is_file():
            self.learn.load(f'{self.name}', purge=False)
            current = self.get_monitor_value()
            print(float(current))
