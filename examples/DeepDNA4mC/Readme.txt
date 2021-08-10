Implementation of "Systematic analysis and accurate identification of DNA N4-methylcytosine sites by deep learning"
===============================

Author
=============================
jingry@scu.edu.cn
ljs@swmu.edu.cn

Files
================================
data:	This folder is the source of all data, C.elegans_Po and C.elegans_Ne in the folder named data are the Zeng_2020_2 dataset, and the data of the remaining three species are the Zeng_2020_1 dataset
model:	This folder is the source of all models.



Usage
================================
Using CNN with onehot encoding£º
```
python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/DeepDNA4mC/data/C.elegans_P.txt examples/DeepDNA4mC/data/C.elegans_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/CNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
```
where both species and model can be selected, and you can change the species and model according to your purpose, by selecting the species dataset in the data folder and the model in the model folder. Here are a few examples: 

Using RNN with onehot encoding:
```
python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/DeepDNA4mC/data/C.elegans_P.txt examples/DeepDNA4mC/data/C.elegans_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/RNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
```

Using CNN-RNN with onehot encoding:
```
python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/DeepDNA4mC/data/C.elegans_P.txt examples/DeepDNA4mC/data/C.elegans_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/CNN-RNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
```

Using CNN-RNN_att-2mer with onehot encoding:
```
python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/DeepDNA4mC/data/C.elegans_P.txt examples/DeepDNA4mC/data/C.elegans_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/CNN-RNN_att-2mer.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --useKMer 1 --KMerNum 2
```

Using CNN-RNN_att-3mer with onehot encoding:
```
python running.py --dataType dna --dataEncodingType onehot --dataTrainFilePaths examples/DeepDNA4mC/data/C.elegans_P.txt examples/DeepDNA4mC/data/C.elegans_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/CNN-RNN_att-3mer.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --useKMer 1 --KMerNum 3
```

Using CNN-RNN_att with dcitionary encoding:
```
python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/DeepDNA4mC/data/C.elegans_P.txt examples/DeepDNA4mC/data/C.elegans_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/CNN-RNN_att_dict.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt
```

Using CNN-RNN_att-2mer with dcitionary encoding:
```
python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/DeepDNA4mC/data/A.thaliana_P.txt examples/DeepDNA4mC/data/A.thaliana_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/CNN-RNN_att_dict-2-mer.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --useKMer 1 --KMerNum 2
```

Using CNN-RNN_att-3mer with dcitionary encoding:
```
python running.py --dataType dna --dataEncodingType dict --dataTrainFilePaths examples/DeepDNA4mC/data/A.thaliana_P.txt examples/DeepDNA4mC/data/A.thaliana_N.txt --dataTrainLabel 1 0 --dataSplitScale 0.9 --modelLoadFile examples/DeepDNA4mC/model/CNN-RNN_att_dict-3-mer.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 41 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --useKMer 1 --KMerNum 3
```
*Note:Users can freely adjust the parameters of the models in the model folder.


UMAP, Deep SHAP, Attention
================================
UMAP:
Using our provided tool for UMAP plotting (the tool is available in root-autoBioSeq/tool):
```
python tool/layerUMAP.py --paraFile tmpOut/parameters.txt --outFigFolder tmpOut --interactive 1
```

Deep SHAP:
Please use the jupyter notebook for detail: 
https://github.com/jingry/autoBioSeqpy/blob/2.0/notebook/An%20Example%20of%20Attention%20Layer%20Plotting%20of%20DeepDNA4mC.ipynb

Attention:
Please use the jupyter notebook for detail: 
https://github.com/jingry/autoBioSeqpy/blob/2.0/notebook/An%20Example%20of%20Attention%20Layer%20Plotting%20of%20DeepDNA4mC.ipynb




