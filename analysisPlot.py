"""
   plot real-time loss-acc curve, ROC curve and PR curve
"""

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.losses = {'batch': [], 'epoch':[]}
        self.accuracy = {'batch': [], 'epoch':[]}
        self.val_loss = {'batch':[],'epoch':[]}
        self.val_acc = {'batch':[],'epoch':[]}

    def on_batch_end(self,batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self,batch,logs={}):
        
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def  loss_plot(self, loss_type, savePath = None, showFig = True, **kwargs):
        if not savePath is None or showFig:
            iters=range(len(self.losses[loss_type]))
            plt.figure()
            plt.plot(iters,self.accuracy[loss_type],'r',label='train acc')
            plt.plot(iters,self.losses[loss_type],'g',label='train loss')
            if loss_type == 'epoch':
                plt.plot(iters, self.val_acc[loss_type],'b',label='val acc')
                plt.plot(iters, self.val_loss[loss_type],'k',label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            if not savePath is None:
                plt.savefig(savePath,**kwargs)
            if showFig:
                plt.show()

def plotROC(test,score, auc=None, savePath = None, showFig = True, **kwargs):
    fpr,tpr,threshold = roc_curve(test, score)
    plt.figure()
    lw = 3
    plt.figure(figsize=(10,10))
    if auc is None:
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
    else:
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=18)
    font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
    plt.xlabel('False Positive Rate',font)
    plt.ylabel('True Positive Rate',font)
    plt.title('Receiver operating characteristic curve',font)
    plt.legend(loc="lower right")
    if not savePath is None:
        plt.savefig(savePath, **kwargs)

    if showFig:
        plt.show()

def plotPR(test,score,savePath = None, showFig = True, **kwargs):
    precision, recall, thresholds = precision_recall_curve(test, score)
    plt.figure()
    lw = 3
    plt.figure(figsize=(10,10))
    plt.plot(precision, recall, color='darkred',lw=lw, label='P-R curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=18)
    font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
    plt.xlabel('Recall',font)
    plt.ylabel('Precision',font)
    plt.title('Precision recallcurve',font)
    plt.legend(loc="lower right")
    if not savePath is None:
        plt.savefig(savePath, **kwargs)

    if showFig:
        plt.show()

    
    