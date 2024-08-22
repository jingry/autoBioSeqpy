EnsembleDL-Lipo
==================
Optimizing Lipocalin Sequence Classification with Ensemble Deep Learning Models
==================

Author
==================
ljs@swmu.edu.cn

Overview
==================
This study employed Convolutional Neural Network (CNN) and Deep Neural Network (DNN) architectures to construct the ensemble framework, EnsembleDL-Lipo, for the precise identification of lipocalins from their primary sequences. The CNN architecture utilized a dictionary encoding method to extract protein sequence information, while the DNN architecture employed nine PSSM-based features to represent protein sequences. A total of 511 unique deep learning models were generated through permutations, and their performance in lipocalin recognition was evaluated, with particular emphasis on the top ten models exhibiting exceptional results. By integrating these individual models with varying input features, we developed the EnsembleDL model, which combines a CNN model with dictionary encoding and a DNN model with three specific PSSM-based features (DFMCA_PSSM, DPC-PSSM, and PSSM-AC). To determine the most effective approach for lipocalin recognition, the performance of a single deep learning model with high prediction accuracy was compared against the ensemble deep learning framework. Our results demonstrate that this ensemble deep learning approach can accurately identify lipocalin proteins and outperform existing methods on the same task.

Usage
==================
We developed a tool, 'generateCMD.py', to identify the optimal ensemble deep learning framework. It is also straightforward to use; simply type 'python examples\EnsembleDL-Lipo\generateCMD.py' under the main autoBioSeqpy path. All results will be stored in the "out" folder within the main path. We found that the ensemble deep learning framework with the 'CNN+DFMCA_PSSM+DPC-PSSM+PSSM-AC' feature group outperformed other combinations. To use this ensemble framework for predicting the independent test set, users can execute the following command:
python running.py --dataType protein other other other --dataEncodingType dict other other other --dataTrainFilePaths examples/Lipo/data/1-212.txt examples/Lipo/data/0-211.txt examples/Lipo/data/po-DFMCA_PSSM.txt examples/Lipo/data/ne-DFMCA_PSSM.txt examples/Lipo/data/po-dpc_pssm.txt examples/Lipo/data/ne-dpc_pssm.txt examples/Lipo/data/po-pssm_ac.txt examples/Lipo/data/ne-pssm_ac.txt --dataTrainLabel 1 0 1 0 1 0 1 0 --dataTesLipoilePaths examples/Lipo/data/positive42-2.txt examples/Lipo/data/negative53.txt examples/Lipo/data/pote-DFMCA_PSSM.txt examples/Lipo/data/nete-DFMCA_PSSM.txt examples/Lipo/data/pote-dpc_pssm.txt examples/Lipo/data/nete-dpc_pssm.txt examples/Lipo/data/pote-pssm_ac.txt examples/Lipo/data/nete-pssm_ac.txt --dataTestLabel 1 0 1 0 1 0 1 0 --modelLoadFile examples/Lipo/model/CNN.py examples/Lipo/model/DFMCA_PSSM.py examples/Lipo/model/dpc_pssm.py examples/Lipo/model/pssm_ac.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 64 --epochs 20 --shuffleDataTrain 1 --spcLen 2000 2000 2000 2000 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0 1 1 2 2 3 3 --dataTestModelInd 0 0 1 1 2 2 3 3    


layerUMAP:
python tool/layerUMAP.py --paraFile tmpOut/parameters.txt --ouLipoigFolder tmpOut --metric cosine --n_neighbors 28 --min_dist 0.8 --interactive 1