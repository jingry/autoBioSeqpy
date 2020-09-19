Gram-negative bacteria secreted proteins classification

Author
=============================
xinyan_scu@126.com
ljs@swmu.edu.cn

Overview
==============================
We use it to apply four trained model to Gram-negative bacteria secreted proteins classification and visualize the resulting predic-tions.

Files
================================
data file
T1SS_train.txt:     112 sequences     (Training dataset)
T1SS_test.txt:     25 sequences     (Independent test dataset)
T2SS_train.txt:     99 sequences     (Training dataset)
T2SS_test.txt:     29 sequences     (Independent test dataset)
T3SS_train.txt:     182 sequences     (Training dataset)
T3SS_test.txt:     28 sequences     (Independent test dataset)
T4SS_train.txt:     62 sequences     (Training dataset)
T4SS_test.txt:     22 sequences     (Independent test dataset)
T5SS_train.txt:     164 sequences     (Training dataset)
T5SS_test.txt:     35 sequences     (Independent test dataset)
T7SS_train.txt:     48 sequences     (Training dataset)
T7SS_test.txt:     33 sequences     (Independent test dataset)

model file
CNN.py: convolutional neural network
RNN.py: recurrent neural network
CNN_RNN.py: convolutional-recurrent neural network
DNN.py: deep neural network


Usage
=================================
When training a deep leanring model, the user can enter the following commands in the command terminal. Take the RNN architecture as an example: 

python running.py --dataType protein --dataEncodingType onehot  --dataTrainFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_train.txt --dataTrainLabel 0 1 2 3 4 5 --dataTestFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_test.txt --dataTestLabel 0 1 2 3 4 5 --modelLoadFile examples/Gramnegativebacterialsecretedproteins/model/CNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 10 --epochs 30 --spcLen 	2000 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --labelToMat 1

For the CNN architecture: 

python running.py --dataType protein --dataEncodingType onehot  --dataTrainFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_train.txt --dataTrainLabel 0 1 2 3 4 5 --dataTestFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_test.txt --dataTestLabel 0 1 2 3 4 5 --modelLoadFile examples/Gramnegativebacterialsecretedproteins/model/CNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 10 --epochs 30 --spcLen 	2000 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --labelToMat 1

For the CNN-RNN architecture: 

python running.py --dataType protein --dataEncodingType onehot  --dataTrainFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_train.txt --dataTrainLabel 0 1 2 3 4 5 --dataTestFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_test.txt --dataTestLabel 0 1 2 3 4 5 --modelLoadFile examples/Gramnegativebacterialsecretedproteins/model/CNN-RNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 10 --epochs 30 --spcLen 	2000 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --labelToMat 1

For the RNN architecture: 
python running.py --dataType protein --dataEncodingType onehot  --dataTrainFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_train.txt --dataTrainLabel 0 1 2 3 4 5 --dataTestFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_test.txt --dataTestLabel 0 1 2 3 4 5 --modelLoadFile examples/Gramnegativebacterialsecretedproteins/model/RNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 10 --epochs 30 --spcLen 	2000 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --labelToMat 1

For the DNN architecture: 
python running.py --dataType protein --dataEncodingType onehot  --dataTrainFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_train.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_train.txt --dataTrainLabel 0 1 2 3 4 5 --dataTestFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T2SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T3SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T4SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T5SS_test.txt examples/Gramnegativebacterialsecretedproteins/data/T7SS_test.txt --dataTestLabel 0 1 2 3 4 5 --modelLoadFile examples/Gramnegativebacterialsecretedproteins/model/DNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 10 --epochs 30 --spcLen 	2000 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --labelToMat 1

Users can refer to the Manual.docx of autoBioSeqpy tool for detailed parameter information. After training, the prediction results, probability, model parameters, weights, optimizer states and generated figures, (including acc-loss curve, ROC curve and PR curve) are all saved in the tmpOut file. Please note that becasue the training set is re-shuffled during each training process, the results may be different each time. When the users want to use the build model to predict the independent test dataset, the following commands are available: 

python predicting.py --paraFile tmpOut/parameters.txt --dataTestFilePaths examples/Gramnegativebacterialsecretedproteins/data/T1SS_test.txt --predictionSavePath tmpout/indPredictions.txt

If users would like to plot the intermediate output from the built model, the following command is available:

python tool/layerUMAP.py --paraFile tmpOut/parameters.txt --outFigFolder tmpOut --metric chebyshev --interactive 1 --theme blue --n_neighbors 4 --min_dist 0.4

Requirements and installation
============================
Please see the README.md of auotBioSeqpy

