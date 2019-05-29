from fastai.tabular import *

class ClassificationInterpretationTabular():
    "Interpretation methods for classification models with tabular data."
    def __init__(self, data:DataBunch, probs:Tensor, y_true:Tensor, losses:Tensor):
        self.data,self.probs,self.y_true,self.losses = data,probs,y_true,losses
        self.pred_class = self.probs.argmax(dim=1)

    @classmethod
    def from_learner(cls, learn:Learner, ds_type:DatasetType=DatasetType.Valid):
        "Create an instance of `ClassificationInterpretationTabular`"
        preds = learn.get_preds(ds_type=ds_type, with_loss=True)
        return cls(learn.data, *preds)

    def confusion_matrix(self):
        "Confusion matrix as an `np.ndarray`."
        x=torch.arange(0,self.data.c)
        cm = torch.zeros(self.data.c, self.data.c, dtype=x.dtype)
        for i in range(0, self.y_true.shape[0], 1):
            cm_slice = ((self.pred_class[i:i+1]==x[:,None])
                        & (self.y_true[i:i+1]==x[:,None,None])).sum(2)
            torch.add(cm, cm_slice, out=cm)
        return to_np(cm)

    def plot_confusion_matrix(self, normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", norm_dec:int=2, **kwargs)->None:
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        cm = self.confusion_matrix()
        plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(self.data.c)
        plt.xticks(tick_marks, list(self.data.classes), rotation=90)
        plt.yticks(tick_marks, list(self.data.classes), rotation=0)

        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
    def most_confused(self, min_val:int=1, slice_size:int=1)->Collection[Tuple[str,str,int]]:
            "Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences."
            cm = self.confusion_matrix()
            np.fill_diagonal(cm, 0)
            res = [(self.data.classes[i],self.data.classes[j],cm[i,j])
                    for i,j in zip(*np.where(cm>=min_val))]
            return sorted(res, key=itemgetter(2), reverse=True)
   
    def plot_top_losses(self, k, largest = True, return_table:bool=False):
        "Shows the respective rows in top_losses along with their prediction, actual, loss, and probability"
        tl_val, tl_idx = self.top_losses(k)
        classes = self.data.classes
        cat_names = self.data.x.cat_names
        cont_names = self.data.x.cont_names
        df = pd.DataFrame(columns = [['Prediction', 'Actual', 'Loss','Prob'] +  cat_names + cont_names])
        for i, idx in enumerate(tl_idx):
          da, cl = self.data.dl().dataset[idx]
          cl = int(cl)
          t1 = str(da)
          t1 = t1.split(';')
          arr = []
          arr.append(classes[self.pred_class[idx]])
          arr.append(classes[cl])
          arr.append(f'{self.losses[idx]:.2f}')
          arr.append(f'{self.probs[idx][cl]:.2f}')
          for x in range(len(t1)-1):
            _, value = t1[x].split()
            arr.append(value)
          df.loc[i] = arr
        display(df)
        if return_table: return df
        
    def top_losses(self, k:int=None, largest=True):
        "`k` largest(/smallest) losses and indexes, defaulting to all losses (sorted by `largest`)."
        return self.losses.topk(ifnone(k, len(self.losses)), largest=largest)
