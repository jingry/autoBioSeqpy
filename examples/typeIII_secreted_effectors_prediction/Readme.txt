Predicting type III secreted effectors and making model comparisons
==============================

Author
=============================
jingry@scu.edu.cn
ljs@swmu.edu.cn

Overview
==============================
We described various deep learning architectures that can be applied to T3SS effectors prediction:
Long Short-Term Memory neural network (LSTM): 
(1)	Embedding layer (embedding size, 256)
(2)	LSTM layer (output size, 64; dropout, 20%, recurrent dropout, 20%)
(3)	Sigmoid output layer
Bidirectional Long Short-Term Memory neural network (biLSTM):
(1)	Embedding layer (embedding size, 256)
(2)	Bidirectional LSTM layer (output size, 64; dropout, 50%)
(3)	Sigmoid output layer
Convolutional neural network (CNN):
(1)	Embedding layer (embedding size, 256, dropout, 20%)
(2)	Convolutional layer (250 kernels; kernel size, 5)
(3)	Pooling layer (Max Pooling, pooling size, 2)
(4)	Flatten layer
(5)	Fully connected layer (650 neural nodes, dropout, 50%)
(6)	Sigmoid output layer
Architecture consisting of Convolutional neural network and Long Short-term memory neural network (CNN-LSTM):
(1)	Embedding layer (embedding size, 256, dropout, 20%)
(2)	Convolutional layer (250 kernels; kernel size, 5)
(3)	Pooling layer (Max Pooling, pooling size, 2)
(4)	LSTM layer (output size, 64)
(5)	Sigmoid output layer
Architecture consisting of Convolutional neural network and Bidirectional Long Short-term memory neural network (CNN-biLSTM):
(1)	Embedding layer (embedding size, 256, dropout, 20%)
(2)	Convolutional layer (250 kernels; kernel size, 5)
(3)	Pooling layer (Max Pooling, pooling size, 2)
(4)	Bidirectional LSTM layer (output size, 64)
(5)	Sigmoid output layer

Files
================================
data file:
train_pos.txt: 303 T3SEs (Training dataset) 
train_neg.txt: 604 non-T3SEs (Training dataset) 
test_pos.txt:  76 T3SEs (Test dataset)
test_neg.txt:  151 non-T3SEs (Test dataset)

model file:

protein_biLSTM_model.py: Bidirectional Long Short-Term Memory neural network 
protein_CNN_model.py: Convolutional neural network 
protein_CNN-biLSTM_model.py: Architecture consisting of Convolutional neural network and Bidirectional Long Short-term memory neural network 
protein_CNN-LSTM_model.py: Architecture consisting of Convolutional neural network and Long Short-term memory neural network 
protein_LSTM_model.py: Long Short-Term Memory neural network 

Usage
=================================
When training a deep leanring model, the user can enter the following commands in the command terminal£º

# protein_biLSTM_model.py

python running.py --dataType protein --dataEncodingType dict  --dataTrainFilePaths examples/typeIIIsecretedeffectors/data/train_pos.txt examples/typeIIIsecretedeffectors/data/train_neg.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/typeIIIsecretedeffectors/data/test_pos.txt examples/typeIIIsecretedeffectors/data/test_neg.txt --dataTestLabel 1 0 --modelLoadFile examples/typeIIIsecretedeffectors/model/protein_biLSTM_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 60 --epochs 20 --spcLen 100 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt 
###########################################################################
# protein_CNN_model.py

python running.py --dataType protein --dataEncodingType dict  --dataTrainFilePaths examples/typeIIIsecretedeffectors/data/train_pos.txt examples/typeIIIsecretedeffectors/data/train_neg.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/typeIIIsecretedeffectors/data/test_pos.txt examples/typeIIIsecretedeffectors/data/test_neg.txt --dataTestLabel 1 0 --modelLoadFile examples/typeIIIsecretedeffectors/model/protein_CNN_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 60 --epochs 20 --spcLen 100 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
###########################################################################
# protein_CNN-biLSTM_model.py 

python running.py --dataType protein --dataEncodingType dict  --dataTrainFilePaths examples/typeIIIsecretedeffectors/data/train_pos.txt examples/typeIIIsecretedeffectors/data/train_neg.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/typeIIIsecretedeffectors/data/test_pos.txt examples/typeIIIsecretedeffectors/data/test_neg.txt --dataTestLabel 1 0 --modelLoadFile examples/typeIIIsecretedeffectors/model/protein_CNN-biLSTM_model.py  --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 60 --epochs 20 --spcLen 100 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
###########################################################################
# protein_LSTM_model.py 

python running.py --dataType protein --dataEncodingType dict  --dataTrainFilePaths examples/typeIIIsecretedeffectors/data/train_pos.txt examples/typeIIIsecretedeffectors/data/train_neg.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/typeIIIsecretedeffectors/data/test_pos.txt examples/typeIIIsecretedeffectors/data/test_neg.txt --dataTestLabel 1 0 --modelLoadFile examples/typeIIIsecretedeffectors/model/protein_LSTM_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 60 --epochs 20 --spcLen 100 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
###########################################################################
# protein_CNN-LSTM_model.py

python running.py --dataType protein --dataEncodingType dict  --dataTrainFilePaths examples/typeIIIsecretedeffectors/data/train_pos.txt examples/typeIIIsecretedeffectors/data/train_neg.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/typeIIIsecretedeffectors/data/test_pos.txt examples/typeIIIsecretedeffectors/data/test_neg.txt --dataTestLabel 1 0 --modelLoadFile examples/typeIIIsecretedeffectors/model/protein_CNN-LSTM_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 60 --epochs 20 --spcLen 100 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt

