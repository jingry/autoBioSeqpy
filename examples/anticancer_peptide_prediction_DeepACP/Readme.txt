DeepACP
=============================
DeepACP: a novel computational approach for accurate identification of anticancer peptides by deep learning algorithm

Author
=============================
xinyan_scu@126.com
ljs@swmu.edu.cn

Overview
==============================
We train several deep neural network-based models to computationally predict anticancer activity from peptide sequence. We design three architectures, namely, convolution neural network (CNN), recurrent neural network (RNN) with bidirectional long short-term memory cells (biLSTM), and hybrid neural network that combine CNN and RNN, which use peptide primary sequence as input, and output a probability score between 0 and 1. More precisely, the input to the models is a sequence of one-letter encoded amino acids (AAs), where each of the 20 basic AAs are assigned a number 1-20 and unknown ¡°X¡± characters are assigned 0 respectively and the output of the models consists of one score which map to the [0, 1] interval, corresponding to the probability of the peptide of interest being an ACP and a non-ACP.Finally,by utilizing the RNN model, we implement a sequence-based deep learning tool, called DeepACP to accurately predict the likelihood of a peptide being presented with anticancer activity. 

Files
================================
data file
ACPs250.txt:     250 anticaner peptides     (Training dataset)
non-ACPs250.txt: 250 non-anticaner peptides (Training dataset)
ACPs82.txt:      82  anticaner peptides     (Independent test dataset)
non-ACPs82.txt   82  non-anticaner peptides (Independent test dataset)
ACPs10.txt:    10 designed peptides with anticaner activities collected from Grisoni et al. (Designing Anticancer Peptides by Constructive Machine Learning. ChemMedChem. 2018, 13(13), 1300-1302)

model file
protein_RNN_model.py: recurrent neural network with bidirectional long short-term memory cells
protein_CNN_model.py: convolution neural network
protein_CNN-RNN_model.py: hybrid neural network that combine CNN and RNN

DeepACP file
parameters.txt: autoBioSeqpy training parameters for the DeepACP
tmpMod.json and tmpWeight.bin: : profiles consisting of the architecture, weights and optimizer state of DeepACP

Usage
=================================
When training a deep leanring model, the user can enter the following commands in the command terminal. Take the RNN architecture as an example: 

python running.py --dataType protein --dataEncodingType dict --dataTrainFilePaths examples/anticancerpeptideprediction/data/ACPs250.txt examples/anticancerpeptideprediction/data/non-ACPs250.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/anticancerpeptideprediction/model/protein_RNN_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 50 --epochs 20 --shuffleDataTrain 1 --spcLen 100 --noGPU 0 --paraSaveName parameters.txt --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin

For the CNN architecture: 

python running.py --dataType protein --dataEncodingType dict --dataTrainFilePaths examples/anticancerpeptideprediction/data/ACPs250.txt examples/anticancerpeptideprediction/data/non-ACPs250.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/anticancerpeptideprediction/model/protein_CNN_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 50 --epochs 20 --shuffleDataTrain 1 --spcLen 100 --noGPU 0 --paraSaveName parameters.txt --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin

For the CNN-RNN architecture: 

python running.py --dataType protein --dataEncodingType dict --dataTrainFilePaths examples/anticancerpeptideprediction/data/ACPs250.txt examples/anticancerpeptideprediction/data/non-ACPs250.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/anticancerpeptideprediction/model/protein_CNN-RNN_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 50 --epochs 20 --shuffleDataTrain 1 --spcLen 100 --noGPU 0 --paraSaveName parameters.txt --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin

Users can refer to the Manual.docx of autoBioSeqpy tool for detailed parameter information. After training, the prediction results, probability, model parameters, weights, optimizer states and generated figures, (including acc-loss curve, ROC curve and PR curve) are all saved in the tmpOut file. Please note that becasue the training set is re-shuffled during each training process, the results may be different each time. When the users want to use the build model to predict the independent test dataset, the following commands are available: 

python predicting.py --paraFile tmpOut/parameters.txt --dataTestFilePaths examples/anticancerpeptideprediction/data/ACPs82.txt --predictionSavePath tmpout/indPredictions.txt

python predicting.py --paraFile tmpOut/parameters.txt --dataTestFilePaths examples/anticancerpeptideprediction/data/non-ACPs82.txt --predictionSavePath tmpout/indPredictions.txt

The prediction results are saved in the indPredictions.txt. As an alternative, users can use the following command to complete model training and independent test set prediction at the same time:

python running.py --dataType protein --dataEncodingType dict --dataTrainFilePaths examples/anticancerpeptideprediction/data/ACPs250.txt examples/anticancerpeptideprediction/data/non-ACPs250.txt --dataTrainLabel 1 0  --dataTestFilePaths examples/anticancerpeptideprediction/data/ACPs82.txt  examples/anticancerpeptideprediction/data/non-ACPs82.txt --dataTestLabel 1 0 --modelLoadFile examples/anticancerpeptideprediction/model/protein_RNN_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 50 --epochs 20 --shuffleDataTrain 0 --spcLen 100 --noGPU 0 --paraSaveName parameters.txt --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin

The difference between the above two commands is that the former splits the training dataset into two parts: 80% of data is used for training, 20% is used for testing(--dataSplitScale 0.8), while the latter uses all data to train the model. Since we choose the RNN model to develop the DeepACP, the model parameters and weights are all save in the DeepACP file. Users first need to transfer the files(parameters.txt, tmpMod.json and tmpWeight.bin) in DeepACP file to the tmpOut file and then use the following commands to predict new data£º

python predicting.py --paraFile tmpOut/parameters.txt --dataTestFilePaths examples/anticancerpeptideprediction/data/ACPs10.txt --predictionSavePath tmpout/DeepACP_result.txt

The prediction results are saved in the DeepACP_result.txt in the tmpOut file.  


Requirements and installation
============================
Please see the README.md of auotBioSeqpy
















