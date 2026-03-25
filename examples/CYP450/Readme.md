DeepCYP450
==================

Accurate identification of cytochrome P450 proteins using multimodal integration of protein language models
==================

Author
==================
ljs@swmu.edu.cn

Overview
==================
DeepCYP450 is a multimodal deep learning framework built on autoBioSeqpy for CYP450 protein identification. It integrates three complementary representations: raw protein sequences processed by convolutional layers, semantic embeddings from pretrained protein language models, and evolutionary features derived from PSSM profiles. Each modality is encoded through separate network branches and fused in a multimodal layer, enabling flexible exploration of feature combinations and efficient batch prediction on large proteomic datasets.

Installation
==================
# Install required packages
pip install -r requirements.txt

Data
==================
NOTE: Please UNZIP the compressed data the first time using:
`7z x data.7z.001`

All datasets used in this study are included in the "data" folder. The folder is organized as follows:
•  Training data (train) – Contains 11 files used for multimodal fusion:
    •1  protein sequence file for training positive and negative samples (trpos.txt, trneg.txt)
    •6  protein language model (PLM) feature files (trpos_*.txt, trneg_*.txt)
    •4 PSSM-derived feature files (trpos_*.txt, trneg_*.txt)
•  Cross-validation data (5fold) – Contains the same type of files as train, organized for 5-fold validation experiments.
•  Test data (test) – Contains only the subset of files required for predictions using the final optimal multimodal combination.
•  Independent application data (indep) – Contains datasets used for applying the model to screened proteins or other external validation sets.

File naming convention:
•  trpos.txt – training set positive sample sequences
•  trneg.txt – training set negative sample sequences
•  trpos_Ankh.txt / trneg_Ankh.txt – corresponding Ankh feature data
•  Other feature files follow the same pattern, representing different PLM or PSSM features.
All datasets include protein sequences and derived feature representations necessary for model training, cross-validation, and final evaluation.

Model
==================
All neural network models used in this study are included in the model folder. The folder contains a total of 11 model files:
• 1 CNN model
• 10 DNN models, each corresponding to a specific input feature type. The DNN model filenames indicate the feature they are trained on.
These models are used for training, cross-validation, and generating predictions for the multimodal fusion framework. Users can directly load these files to reproduce model results or perform batch predictions.

Usage
==================
You can directly run the generateCMD-Train.py file in the root directory to execute multimodal fusion tests. In this process, 1023 model combinations are tested, each repeated five times. All results are stored in the result/1023 folder. Users can access these files to analyze model performance, explore modality combinations, or reproduce experiments.
>> python generateCMD-Train.py

Optimal Model Training and Independent Test
==================
Among the 1023 multimodal combinations, the combination of CNN (One-hot), DNN (ESM-1b), DNN (ESM-2), DNN (ProtT5), and DNN (Ankh) achieved the best performance. These five modules constitute the final DeepCYP450 model structure. To train the optimal model combination and perform prediction on the independent test set, run the following command from the root directory:

>> python running.py --dataType protein other other other other --dataEncodingType onehot other other other other --dataTrainFilePaths ./data/train/trpos.txt ./data/train/trneg.txt ./data/train/trpos_ESM1b.txt ./data/train/trneg_ESM1b.txt ./data/train/trpos_ESM2.txt ./data/train/trneg_ESM2.txt ./data/train/trpos_T5.txt ./data/train/trneg_T5.txt ./data/train/trpos_Ankh.txt ./data/train/trneg_Ankh.txt --dataTrainLabel 1 0 1 0 1 0 1 0 1 0 --dataTestFilePaths ./data/test/tepos.txt ./data/test/teneg.txt ./data/test/tepos_ESM1b.txt ./data/test/teneg_ESM1b.txt ./data/test/tepos_ESM2.txt ./data/test/teneg_ESM2.txt ./data/test/tepos_T5.txt ./data/test/teneg_T5.txt ./data/test/tepos_Ankh.txt ./data/test/teneg_Ankh.txt --dataTestLabel 1 0 1 0 1 0 1 0 1 0  --outSaveFolderPath tmpOut --showFig True --saveFig True --modelLoadFile ./model/CNN.py ./model/ESM1b.py ./model/ESM2.py ./model/T5.py ./model/Ankh.py --shuffleDataTrain 1 --batch_size 60 --epochs 20 --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0 1 1 2 2 3 3 4 4 --dataTestModelInd 0 0 1 1 2 2 3 3 4 4 --spcLen 1000 1000 1000 1000 1000

All results will be stored in the result/tmpOut folder.

5-Fold Cross-Validation
===================
To perform 5-fold cross-validation, run the running.py script with the corresponding fold data. The command for fold0 is as follows:

>>python running.py --dataType protein other other other other --dataEncodingType onehot other other other other --dataTrainFilePaths ./data/5fold/fold0/trpos.txt ./data/5fold/fold0/trneg.txt ./data/5fold/fold0/trpos_ESM1b.txt ./data/5fold/fold0/trneg_ESM1b.txt ./data/5fold/fold0/trpos_ESM2.txt ./data/5fold/fold0/trneg_ESM2.txt ./data/5fold/fold0/trpos_T5.txt ./data/5fold/fold0/trneg_T5.txt ./data/5fold/fold0/trpos_Ankh.txt ./data/5fold/fold0/trneg_Ankh.txt --dataTrainLabel 1 0 1 0 1 0 1 0 1 0 --dataTestFilePaths ./data/5fold/fold0/tepos.txt ./data/5fold/fold0/teneg.txt ./data/5fold/fold0/tepos_ESM1b.txt ./data/5fold/fold0/teneg_ESM1b.txt ./data/5fold/fold0/tepos_ESM2.txt ./data/5fold/fold0/teneg_ESM2.txt ./data/5fold/fold0/tepos_T5.txt ./data/5fold/fold0/teneg_T5.txt ./data/5fold/fold0/tepos_Ankh.txt ./data/5fold/fold0/teneg_Ankh.txt --dataTestLabel 1 0 1 0 1 0 1 0 1 0  --outSaveFolderPath tmpOut --showFig True --saveFig True --modelLoadFile ./model/CNN.py ./model/ESM1b.py ./model/ESM2.py ./model/T5.py ./model/Ankh.py --shuffleDataTrain 1 --batch_size 60 --epochs 20 --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0 1 1 2 2 3 3 4 4 --dataTestModelInd 0 0 1 1 2 2 3 3 4 4 --spcLen 1000 1000 1000 1000 1000

To run other folds, simply replace fold0 with fold1, fold2, fold3, or fold4 in all file paths. All cross-validation results will be saved in the specified tmpOut folder.

Other Notes
===================
For detailed explanations of the command-line parameters used in the scripts, please refer to the manual.docx file.

The complete code, datasets, and results are available at zenodo (https://zenodo.org/records/18180828).



