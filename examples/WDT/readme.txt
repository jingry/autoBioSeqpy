 
Identification of thrombopoiesis inducer based on a hybrid deep neural network model

Author
=============================
ljs@swmu.edu.cn
moqi17@sina.cn

Overview
==============================
Thrombocytopenia is a commonly encountered hematologic problem worldwide. At present, there are no relatively safe and effective agents to treat thrombocytopenia. To address this challenge, we proposed a computational method that enables the discovery of novel drug candidates with hematopoietic activity. Based on different types of molecular representations, three deep learning (DL) algorithms, including recurrent neural networks (RNNs), deep neural networks (DNNs), and hybrid neural networks (RNNs+DNNs), were used to develop classification models to distinguish between active and inactive compounds. The evaluation results illustrated that the hybrid DL model exhibited the best prediction performance with 97.8% accuracy and 0.958 MCC on the test dataset. Subsequently, we performed a drug discovery screening based on the hybrid DL model and identified a compound from the FDA-approved drug library that was structurally divergent from conventional drugs and showed a potential therapeutic action on thrombocytopenia. We reported that the new drug candidate wedelolactone significantly promoted megakaryocyte differentiation in vitro, increased platelet level and megakaryocyte differentiation in irradiated mice with no system toxicity. Overall, our work represents an example of how artificial intelligence can be used to discover novel drugs against thrombocytopenia. 

Files
================================
data file
pos_train.txt: 255  active compounds   (Training dataset)
neg_train.txt: 299 inactive compounds  (Training dataset)
pos_train_feature.txt: Molecular fingerprints of 255  active compounds  (Training dataset)
neg_train_feature.txt: Molecular fingerprints of 299 inactive compounds  (Training dataset)
FDA-Library.txt: 817 compounds of unknown activity
FDA-Library_feature.txt: Molecular fingerprints of 817 compounds of unknown activity

model file
BiGRU.py: recurrent neural network with bidirectional gating recurrent unit
BiGRU_hybrid.py: recurrent neural network with bidirectional gating recurrent unit
DNN.py: dynamic neural network
DNN_hybrid.py: dynamic neural network

Usage
=================================
When training a deep leanring model, the user can enter the following commands in the command terminal. Take the RNN architecture as an example: 

For the RNN architecture: 

Python running.py --dataType smiles --dataEncodingType onehot --dataTrainFilePaths examples/WDT/data/pos_train.txt examples/WDT/data/neg_train.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/WDT/model/BiGRU.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 350 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False)


For the DNN architecture: 

python running.py --dataType other --dataEncodingType other --dataTrainFilePaths examples/WDT/data/pos_train_feature.txt examples/WDT/data/neg_train_feature.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/WDT/model/DNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 2048 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False)


====================================================================

For the Hybrid RNN+DNN architecture:

python running.py --dataType smiles other --dataEncodingType onehot other --dataTrainFilePaths examples/WDT/data/pos_train.txt examples/WDT/data/neg_train.txt examples/WDT/data/pos_train_feature.txt examples/WDT/data/neg_train_feature.txt --dataTrainLabel 1 0 1 0 --dataSplitScale 0.8 --modelLoadFile examples/WDT/model/BiGRU_hybrid.py examples/WDT/model/DNN_hybrid.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 32 --epochs 20 --shuffleDataTrain 1 --spcLen 350 2048 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 1 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0 1 1

Users can refer to the Manual.docx of autoBioSeqpy tool for detailed parameter information. After training, the prediction results, probability, model parameters, weights, optimizer states and generated figures, (including acc-loss curve, ROC curve and PR curve) are all saved in the tmpOut file. Please note that becasue the training set is re-shuffled during each training process, the results may be different each time. When the users want to use the build model to predict the dataset, the following commands are available: 

python predicting.py --paraFile tmpOut/parameters.txt --dataTestFilePaths examples/WDT/data/FDA-Library.txt examples/WDT/data/FDA-Library_feature.txt --predictionSavePath tmpout/indPredictions.txt --dataTestModelInd 0 1

The prediction results are saved in the indPredictions.txt.

To validate these results and further understand the internal mechanisms of the DL model, we can use the following command-line command:

python tool/layerUMAP.py --paraFile tmpOut/parameters.txt --outFigFolder tmpOut --interactive 1

Requirements and installation
============================
Please see the README.md of auotBioSeqpy
















