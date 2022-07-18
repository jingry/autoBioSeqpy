Deepm5U
===========================================================================
Evaluation and development of deep neural networks for RNA 5-Methyluridine classifications using autoBioSeqpy


Author
===========================================================================
jingry@scu.edu.cn
ljs@swmu.edu.cn

Files
==========================================================================
data: This folder is the source of all datasets, including 24 datasets, details are described in the paper
model: This folder is the source of all deep learnig models, including CNN, BiGRU, BiLSTM, CNN-BiGRU and CNN-BiLSTM.

Usage
===========================================================================

Model training and prediction, corresponding to instruction 1 in the article
#######################################################################
python running.py --dataType rna --dataEncodingType onehot --dataTrainFilePaths examples/m5U/data/Full_train_positive.txt examples/m5U/data/Full_train_negative.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/m5U/model/CNN-BiLSTM.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False)


Independent test set prediction, corresponding to instruction 3 in the article
########################################################################
python running.py --dataType rna --dataEncodingType onehot --dataTrainFilePaths examples/m5U/data/Full_train_positive.txt  examples/m5U/data/Full_train_negative.txt  --dataTrainLabel 1 0 --dataTestFilePaths examples/m5U/data/Full_test_positive.txt  examples/m5U/data/Full_test_negative.txt  --dataTestLabel 1 0 --modelLoadFile examples/m5U/model/CNN-BiLSTM.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False)


Users can freely replace data sets and models according to their needs
########################################################################

UMAP visualization corresponds to instruction 4 in the article

python tool/layerUMAP.py --paraFile tmpOut/parameters.txt --outFigFolder tmpOut --interactive 1

python tool/layerUMAP.py --paraFile tmpOut/parameters.txt --outFigFolder tmpOut --metric cosine --n_neighbors 28 --min_dist 0.8 --interactive 1



###########################################################################
jupyter notebook

DeepSHAP
Please use the jupyter notebook for detail: 
https://github.com/jingry/autoBioSeqpy/blob/2.0/notebook/Understanding%20the%20contributions%20from%20the%20inputs%20using%20shaps%20(onehot%20case).ipynb

Mutation Plotting
Please use the jupyter notebook for detail: 
https://github.com/jingry/autoBioSeqpy/blob/2.0/notebook/An%20Example%20of%20Mutation%20Plotting.ipynb

We provide a detailed tutorial in the article, please refer to the instructions in the article



