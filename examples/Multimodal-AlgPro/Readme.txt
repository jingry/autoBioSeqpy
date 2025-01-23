Multimodal-AlgPro
=================
Multimodal deep learning for allergenic proteins prediction
=================

Author
==================
ljs@swmu.edu.cn


Overview
==================

Multimodal-AlgPro is an advanced deep learning framework designed to enhance allergen prediction by integrating multimodal data sources. By combining sequence, composition, physicochemical properties, and evolutionary information, it delivers robust and accurate allergen predictions. The model employs a character-level convolutional neural network (CNN) to encode amino acid sequences and a densely connected network (DNN) to extract deeper insights from molecular features. Through systematic evaluation of 12 distinct unimodal models (CNN, AAC, AAC_PSSM, CTD, DC, DFMCA_PSSM, DP_PSSM,DPC_PSSM, Pse_PSSM, PSSM_AC,PSSM400, Single_Average) and 2,047 multimodal combinations (Table S1), we identified the most successful configuration: 'CNN + DFMCA_PSSM + DPC_PSSM + PSSM400 + Single_Average'. This multimodal combination achieved outstanding performance, with average accuracy (93.1%), F-value (92.9%), recall (90.9%), precision (95.1%), MCC (0.863), auROC (0.976), and auPR (0.977). By considering complementary information from five modalities, the framework provides a comprehensive and highly effective approach to allergen prediction.

Usage
==================
You can directly run the generateCMD.py file in the root directory to execute multimodal fusion tests. In this process, 2,047 model combinations are tested, each repeated five times. All results are stored in the "outs" folder, and a comprehensive summary of these results is provided in Table S1.csv, which contains data for all 2,047 combinations.

Command line for Multimodal-AlgPro multimodal training and prediction on independent test set
==================
python running.py --dataType protein other other other other --dataEncodingType onehot other other other other --dataTrainFilePaths ./examples/EnsembleDL-AlgPro/data/trpos2840.txt ./examples/EnsembleDL-AlgPro/data/trneg2840.txt ./examples/EnsembleDL-AlgPro/data/trpo-DFMCA_PSSM.txt ./examples/EnsembleDL-AlgPro/data/trne-DFMCA_PSSM.txt ./examples/EnsembleDL-AlgPro/data/trpo-dpc_pssm.txt ./examples/EnsembleDL-AlgPro/data/trne-dpc_pssm.txt ./examples/EnsembleDL-AlgPro/data/trpo-pssm400.txt ./examples/EnsembleDL-AlgPro/data/trne-pssm400.txt ./examples/EnsembleDL-AlgPro/data/trpo-single_Average.txt ./examples/EnsembleDL-AlgPro/data/trne-single_Average.txt --dataTrainLabel 1 0 1 0 1 0 1 0 1 0 --dataTestFilePaths ./examples/EnsembleDL-AlgPro/data/tepos710.txt ./examples/EnsembleDL-AlgPro/data/teneg710.txt ./examples/EnsembleDL-AlgPro/data/tepo-DFMCA_PSSM.txt ./examples/EnsembleDL-AlgPro/data/tene-DFMCA_PSSM.txt ./examples/EnsembleDL-AlgPro/data/tepo-dpc_pssm.txt ./examples/EnsembleDL-AlgPro/data/tene-dpc_pssm.txt ./examples/EnsembleDL-AlgPro/data/tepo-pssm400.txt ./examples/EnsembleDL-AlgPro/data/tene-pssm400.txt ./examples/EnsembleDL-AlgPro/data/tepo-single_Average.txt ./examples/EnsembleDL-AlgPro/data/tene-single_Average.txt --dataTestLabel 1 0 1 0 1 0 1 0 1 0 --modelLoadFile ./examples/EnsembleDL-AlgPro/model/CNN.py ./examples/EnsembleDL-AlgPro/model/DFMCA_PSSM.py ./examples/EnsembleDL-AlgPro/model/dpc_pssm.py ./examples/EnsembleDL-AlgPro/model/pssm400.py ./examples/EnsembleDL-AlgPro/model/single_Average.py --verbose 1 --showFig 0 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 256 --epochs 20 --shuffleDataTrain 1 --spcLen 1000 1000 1000 1000 1000 --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0 1 1 2 2 3 3 4 4  --dataTestModelInd 0 0 1 1 2 2 3 3 4 4 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin

UMAP Visualization
==================
python tool/layerUMAP.py --paraFile tmpOut/parameters.txt --outFigFolder tmpOut --metric cosine --n_neighbors 28 --min_dist 0.8 --interactive 1


Single model command
==================
python running.py --dataType protein --dataEncodingType onehot --dataTrainFilePaths examples/EnsembleDL-AlgPro/data/trpos2840.txt examples/EnsembleDL-AlgPro/data/trneg2840.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/EnsembleDL-AlgPro/model/CNN.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --showFig 0 --saveFig 1 --batch_size 256 --epochs 20 --shuffleDataTrain 1 --spcLen 2000 --noGPU 1 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin


python running.py --dataType other --dataEncodingType other --dataTrainFilePaths examples/EnsembleDL-AlgPro/data/trpo-pssm_ac.txt examples/EnsembleDL-AlgPro/data/trne-pssm_ac.txt --dataTrainLabel 1 0 --dataSplitScale 0.8 --modelLoadFile examples/EnsembleDL-AlgPro/model/pssm_ac.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --showFig 0 --saveFig 1 --batch_size 256 --epochs 20 --shuffleDataTrain 1 --spcLen 1000 --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin


(*Note that the single model CNN or DNN should include a sigmoid activation function in the output layer)











  

 