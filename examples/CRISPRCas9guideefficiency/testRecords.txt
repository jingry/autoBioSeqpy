Predicting CRISPR/Cas9 guide efficiency and making encoding ways comparisons
===============================

Author
=============================
jingry@scu.edu.cn
ljs@swmu.edu.cn

Overview
==============================
We highlight advancements in the field of sgRNAs design and discuss various encoding ways, which predicts the targeting activity of sgRNAs and improves the performance of deep learning models. For the deep learning model, we chose a framework based on convolutional neural network proposed by Kim et al. The CNN model consisted of a convolution layer, a pooling layer and three fully connected layers. The first convolutional layer performed one-dimensional convolution operations with 80 filters of length 5. After convolution, the rectified linear units (ReLU) was used to output the filter scanning results that were above the thresholds, which was learned during model training. The second pooling layer performed the average in each of the non-overlapping windows of size 2. All the pooling results were then combined in one vector. Three fully connected layers were employed with 80, 40, and 40 units, respectively. A dropout rate of 0.3 was added between each fully connected layer to improve the generalization capability of the model and avoid overfitting. The final output from the CNN is passed through a sigmoid function instead of original regression function, so that predictions were scaled between 0 and 1.

Files
================================
data file:
Doench_high_activity_sgRNA.txt: 368 sgRNAs
Doench_low_activity_sgRNA.txt:  368 sgRNAs
Moreno_high_activity_sgRNA.txt: 204 sgRNAs
Moreno_low_activity_sgRNA.txt:  204 sgRNAs
Wang-Xu_high_activity_sgRNA.txt: 415 sgRNAs
Wang-Xu_low_activity_sgRNA.txt:  415 sgRNAs

model file:
DNA_2mer_CNN1D_model.py: CNN model with k-mer (k=2) embedding representation
DNA_2mer_CNN2D_model.py: CNN model with k-mer (k=2) one-hot encoding
DNA_3mer_CNN1D_model.py: CNN model with k-mer (k=3) embedding representation
DNA_3mer_CNN2D_model.py: CNN model with k-mer (k=3) one-hot encoding
DNA_CNN1D_model.py:      CNN model with k-mer (k=1) embedding representation
DNA_CNN2D_model.py:      CNN model with k-mer (k=1) one-hot encoding

Usage
=================================
When training a deep leanring model, the user can enter the following commands in the command terminal£º

# DNA_CNN2D_model.py

python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/CRISPRCas9guideefficiency/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9guideefficiency/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9guideefficiency/model/DNA_CNN2D_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --firstKernelSize 4 5 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
############################################################################ 
#DNA_2mer_CNN2D_model.py

python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/CRISPRCas9guideefficiency/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9guideefficiency/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9guideefficiency/model/DNA_CNN2D_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --firstKernelSize 16 5 --useKMer 1 --KMerNum 2 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
############################################################################ 
#DNA_3mer_CNN2D_model.py

python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/CRISPRCas9guideefficiency/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9guideefficiency/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9guideefficiency/model/DNA_3mer_CNN2D_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --firstKernelSize 64 5 --useKMer 1 --KMerNum 3 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
###########################################################################
# DNA_CNN1D_model.py

python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/CRISPRCas9guideefficiency/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9guideefficiency/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9guideefficiency/model/DNA_CNN1D_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt 
###########################################################################
# DNA_2mer_CNN1D_model.py

python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/CRISPRCas9guideefficiency/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9guideefficiency/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9guideefficiency/model/DNA_2mer_CNN1D_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --useKMer 1 --KMerNum 2 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
############################################################################ 
#DNA_3mer_CNN1D_model.py

python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/CRISPRCas9guideefficiency/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9guideefficiency/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9guideefficiency/model/DNA_3mer_CNN1D_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --useKMer 1 --KMerNum 3 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
