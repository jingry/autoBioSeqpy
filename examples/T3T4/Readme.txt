DeepT3_4
==================================================================
DeepT3_4: A Hybrid Deep Neural Network Model for the Distinction Between Bacterial Type III and IV Secreted Effectors               
                                                                  
Author                                                            
==================================================================
xinyan_scu@126.com                                                
jingry@scu.edu.cn                                                 

Introduction                                                                
==================================================================
We explore the use of various deep learning architectures and feature descriptors to identify and classify T3SEs and T4SEs. Four different deep learning architectures are used to build the classification models, including the convolutional neural networks (CNNs), recurrent neural networks (RNNs), convolutional-recurrent neural networks (CNN-RNNs) and deep neural networks (DNNs). For the CNN, RNN and CNN-RNN architectures, we first characterize protein sequences using dictionary encoding and then generate amino-acid character embedding vectors to learn the features of two types of secreted effectors. The DNN architecture is designed as a multilayered neural network, whose input layer is fed traditional features or descriptors, including amino acid composition (AAC), dipeptide composition (DC), position specific scoring matrix (PSSM), and their different combinations. We carry out extensive experiments for comparison and present a systematic analysis. Our results show that a hybrid neural network (architectures: RNN + DNN; features: dictionary encoding + AAC + DC) performs better than other models on the training and test datasets, enabling accurate classification of T3SEs and T4SEs. We also achieve interpretable deep learning for T3SEs and T4SEs classification via an advanced dimensionality reduction procedure and visualization, which unravels the predictions of models. Based on these results, we develop a deep learning approach, which is called DeepT3_4, by implementing both the raw sequence and sequence-derived features of effector proteins into the hybrid model.

Files
==================================================================
The dataset file contains benchmark and independent test datasets.

The feature file contains traditional features of proteins, including amino acid composition (AAC), dipeptide composition (DC), position specific scoring matrix (PSSM), and their different combinations.

The model file contains various deep neural networks, inclduing CNN, RNN, CNN-RNN, DNN and the hybrid model integrating RNN and DNN.

Usage
==================================================================
Users can use the following commands to complete model training and test dataset prediction. Using the RNN architecture as an example:

python running.py --dataType protein --dataEncodingType dict --dataTrainFilePaths examples/T3T4/dataset/trainT3.txt examples/T3T4/dataset/trainT4.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/T3T4/dataset/testT3.txt examples/T3T4/dataset/testT4.txt --dataTestLabel 1 0 --modelLoadFile examples/T3T4/model/RNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --showFig 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --spcLen 1500

For the CNN architecture:

python running.py --dataType protein --dataEncodingType dict --dataTrainFilePaths examples/T3T4/dataset/trainT3.txt examples/T3T4/dataset/trainT4.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/T3T4/dataset/testT3.txt examples/T3T4/dataset/testT4.txt --dataTestLabel 1 0 --modelLoadFile examples/T3T4/model/CNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --showFig 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --spcLen 1500

For the CNN-RNN architecture:

python running.py --dataType protein --dataEncodingType dict --dataTrainFilePaths examples/T3T4/dataset/trainT3.txt examples/T3T4/dataset/trainT4.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/T3T4/dataset/testT3.txt examples/T3T4/dataset/testT4.txt --dataTestLabel 1 0 --modelLoadFile examples/T3T4/model/CNN-RNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --showFig 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --spcLen 1500

For the DNN architecture, the user should be aware of the dimensions of input features. Here, we give an example of the combination of ACC and DC:

python running.py --dataType other --dataEncodingType other --dataTrainFilePaths examples/T3T4/feature/trainT3_AAC_DC.txt examples/T3T4/feature/trainT4_AAC_DC.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/T3T4/feature/testT3_AAC_DC.txt examples/T3T4/feature/testT4_AAC_DC.txt --dataTestLabel 1 0 --modelLoadFile examples/T3T4/model/DNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --showFig 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --spcLen 1500

For the final hybrid model, DeepT3_4:

python running.py --dataType protein other --dataEncodingType dict other --dataTrainFilePaths examples/T3T4/dataset/trainT3.txt examples/T3T4/dataset/trainT4.txt examples/T3T4/feature/trainT3_AAC_DC.txt examples/T3T4/feature/trainT4_AAC_DC.txt --dataTrainLabel 1 0 1 0 --dataTestFilePaths examples/T3T4/dataset/testT3.txt examples/T3T4/dataset/testT4.txt examples/T3T4/feature/testT3_AAC_DC.txt examples/T3T4/feature/testT4_AAC_DC.txt --dataTestLabel 1 0 1 0 --modelLoadFile examples/T3T4/model/RNN_hybrid.py examples/T3T4/model/DNN_hybrid.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --showFig 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --spcLen 1500 1500  --dataTrainModelInd 0 0 1 1  --dataTestModelInd 0 0 1 1

When the users only want to train a model, they can use the following commands:

python running.py --dataType protein other --dataEncodingType dict other --dataTrainFilePaths examples/T3T4/dataset/trainT3.txt examples/T3T4/dataset/trainT4.txt examples/T3T4/feature/trainT3_AAC_DC.txt examples/T3T4/feature/trainT4_AAC_DC.txt --dataTrainLabel 1 0 1 0 --dataSplitScale 0.8 --modelLoadFile examples/T3T4/model/RNN_hybrid.py examples/T3T4/model/DNN_hybrid.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --showFig 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --spcLen 1500 1500  --dataTrainModelInd 0 0 1 1  --dataTestModelInd 0 0 1 1

The training dataset was splited into two parts: 80% of data for training, and 20% for testing (--dataSplitScale 0.8). Finally, when the model has been trained and tested, the user can visualize the model with the following commands:

python tool/layerPlot.py --paraFile tmpOut/parameters.txt --outFigFolder tmpOut

The figure is saved in the tmpOut file UMAP.pdf











