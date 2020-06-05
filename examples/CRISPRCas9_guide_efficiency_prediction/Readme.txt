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
DNA_2mer_dict_model.py:   CNN model with k-mer (k=2) dictionary encoding 
DNA_2mer_onehot_model.py£ºCNN model with k-mer (k=2) one-hot encoding
DNA_3mer_dict_model.py£º CNN model with k-mer (k=3) dictionary encoding
DNA_3mer_onehot_model.py£º CNN model with k-mer (k=3) one-hot encoding
DNA_dict_model.py£º CNN model with dictionary encoding
DNA_onehot_model.py£ºCNN model with one-hot encoding 



Usage
=================================
When training a deep leanring model, the user can enter the following commands in the command terminal£º

#DNA_2mer_dict_model.py

python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9_guide_efficiency_prediction/model/DNA_2mer_dict_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --useKMer 1 --KMerNum 2 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
===========================================================================
#DNA_2mer_onehot_model.py

python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9_guide_efficiency_prediction/model/DNA_2mer_onehot_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --useKMer 1 --KMerNum 2 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
===========================================================================
#DNA_3mer_dict_model.py

python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9_guide_efficiency_prediction/model/DNA_3mer_dict_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --useKMer 1 --KMerNum 3 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
===========================================================================
#DNA_3mer_onehot_model.py

python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_high_activity_sgRNA.txt examples/CRISPRCas9_guide_efficiency_prediction/data/Doench_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9_guide_efficiency_prediction/model/DNA_3mer_onehot_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --useKMer 1 --KMerNum 3 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
===========================================================================#DNA_dict_model.py

python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/CRISPRCas9_guide_efficiency_prediction/data/Moreno_high_activity_sgRNA.txt examples/CRISPRCas9_guide_efficiency_prediction/data/Moreno_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9_guide_efficiency_prediction/model/DNA_dict_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt 
===========================================================================
#DNA_onehot_model.py

python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/CRISPRCas9_guide_efficiency_prediction/data/Moreno_high_activity_sgRNA.txt examples/CRISPRCas9_guide_efficiency_prediction/data/Moreno_low_activity_sgRNA.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/CRISPRCas9_guide_efficiency_prediction/model/DNA_onehot_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 25 --epochs 40 --shuffleDataTrain 1 --spcLen 30 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt 
