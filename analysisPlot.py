"""
   plot real-time loss-acc curve, ROC curve and PR curve
"""

import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plotROC(test,score, savePath = None, showFig = True, **kwargs):
    fpr,tpr,threshold = roc_curve(test, score)
    auc_roc = auc(fpr,tpr)
    plt.figure()
    font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22,
         }
    lw = 3
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %auc_roc)
#    if aucVal is None:
#        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
#    else:
#        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %aucVal)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=20)
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
    pr_auc = auc(recall,precision)
    plt.figure()
    lw = 3
    font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22,
         }
    plt.figure(figsize=(8,8))
    plt.plot(precision, recall, color='darkred',lw=lw, label='P-R curve (area = %0.2f)' %pr_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=20)
    plt.xlabel('Recall',font)
    plt.ylabel('Precision',font)
    plt.title('Precision recall curve',font)
    plt.legend(loc="lower right")
    if not savePath is None:
        plt.savefig(savePath, **kwargs)

    if showFig:
        plt.show()

def showMatWithVal(matIn,figSize = (16,10), fontsize = 10, precision = '%.3f', \
                   vmin = 0.0000001, xtitle = None, xticks = None,\
                   xtickLabels=None, ytitle = None, yticks = None,\
                   ytickLabels=None, colorBarTitle = None, xlim = None, \
                   ylim = None, toInvert = True, saveFig = None, saveDpi = 300,\
                   showText = True, color_bar_set_under = (0,0,0,0), vmax=None,\
                   color_bar_set_over = None, cmapName = 'jet',extent=None,
                   stick_size = None, title_size = None, norm=None, 
                   driverCaxSize=0.05, driverCaxPad=0.05):
    data = matIn.copy()
    my_cmap = mpl.cm.get_cmap(cmapName) 
    if not color_bar_set_under is None:
        my_cmap.set_under(color_bar_set_under)
    if not color_bar_set_over is None:
        my_cmap.set_over(color_bar_set_over)
    x_start = 0
    x_end = matIn.shape[1]
    y_start = 0
    y_end = matIn.shape[0]
    size_x = matIn.shape[1]
    size_y = matIn.shape[0]
    if extent is None:
        extent = [x_start, x_end, y_start, y_end]
#    print 'extent:',extent
    # The normal figure
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(111)
    #im = ax.imshow(data, extent=extent, origin='lower', interpolation='None', cmap='viridis')
    

    
    im = ax.imshow(data, extent=extent, origin='lower', interpolation='None', cmap=my_cmap, vmin = vmin, vmax=vmax, norm=norm)

    # Add the text
    jump_x = (x_end - x_start) / (size_x)
    jump_y = (y_end - y_start) / (size_y)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size_x, endpoint=False) - 0.5
    y_positions = np.linspace(start=y_start, stop=y_end, num=size_y, endpoint=False) - 0.5
    
    if showText:
        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
    #             if y_index > data.shape[0] - 1 or x_index > data.shape[1] - 1:
    #                 continue
                label = precision %data[y_index, x_index]
                text_x = x + jump_x
                text_y = y + jump_y
    #             if data[y_index, x_index] > 0:
    #                 print text_x, text_y, label
                if data[y_index, x_index] < vmin:
                    fontColor = 'white'
                elif data[y_index, x_index] < 0:
    #                 fontColor = im.cmap(np.abs(data[x_index, y_index]/data.max()))                
    #                 print label,np.abs(data[y_index, x_index]/data.max()),fontColor
                    r,g,b,a = im.cmap(np.abs(data[y_index, x_index]/data.max()))        
                    fontColor = (1. - r, 1. - g, 1. - b, a)
                else:
                    r,g,b,a = im.cmap(data[y_index, x_index]/data.max())
                    fontColor = (1. - r, 1. - g, 1. - b, a)
    #                 fontColor = im.cmap(1 - data[y_index, x_index]/data.max())
                ax.text(text_x, text_y, label, color=fontColor, ha='center', va='center', fontsize=fontsize )
    
    
    divider = make_axes_locatable(ax)
    caxDriver = divider.append_axes("right", size=driverCaxSize, pad=driverCaxPad)
    cax = fig.colorbar(im,cax = caxDriver)
    if not colorBarTitle is None:
        cax.set_label(colorBarTitle)
    if not xticks is None:
        ax.set_xticks(xticks)
    else:
        xstic = range(x_start,x_end)
        ax.set_xticks(np.array(xstic) + 0.5)
    if not xtickLabels is None:
        ax.set_xticklabels(xtickLabels)
    else:
        xstic = range(x_start,x_end)
        ax.set_xticklabels(np.array(xstic)+1)
    if not xtitle is None:
        if title_size:
            plt.xlabel(xtitle, fontsize = title_size)
        else:
            plt.xlabel(xtitle)
    if not ytitle is None:
        if title_size:
            plt.ylabel(ytitle, fontsize = title_size)
        else:
            plt.ylabel(ytitle)
    #ax.set_xlabel(keyName)  
    if not yticks is None:
        ax.set_yticks(yticks)
    else:
        y_tick = range(y_start,y_end)
        ax.set_yticks(np.array(y_tick) + 0.5)  
    if not ytickLabels is None:
        ax.set_yticklabels(ytickLabels)
    else:
        y_tick = range(y_start,y_end)
        ax.set_yticklabels(np.array(y_tick)+1)
    if not stick_size is None:
        for item in ax.get_xticklabels() + ax.get_yticklabels() + cax.ax.get_yticklabels():
            item.set_fontsize(stick_size)
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)
    if toInvert:
        plt.gca().invert_yaxis()
    if saveFig:
        plt.savefig(saveFig,dpi=saveDpi)
    plt.show()    
    