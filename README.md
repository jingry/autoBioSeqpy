![image](https://github.com/jingry/autoBioSeqpy/blob/master/logo.jpg)
# autoBioSeqpy
a deep learning tool for the classification of biological sequences

# 1.	Brief Introduction 
This document is a guide for users to get touch with our tool ‘autoBioSeqpy’ for modeling and analysis of Protein, DNA and RNA data, including a brief introduction to installation, quick start and standalone modules (a jupyter notebook example provided).
Our tool autoBioSeqpy’ is a self-made python tool which can transfer the sequence into matrix, and then use it for deep learning. In this document, users can find the data format, the internal mechanism of encoding and figure out how to use this tool.

# 2.	Installation 
All the code of autoBioSeqpy’ is wrote in Python, and no mixture code (e.g. C/C++) is used in this project, so the installation is very easy. Once the dependencies are resolved, the only thing to do is to make the path as a working path or put the code into the search path.

A basic way is using

```pip install -r requirements.txt```

 or
 
```pip3 install -r requirements.txt```

is enough. But if users want to install using anaconda or other way manually, please read the followwing subsections.

## 2.1 Dependence
Some python modules are necessary for autoBioSeqpy, which are **re, numpy, importlib, matplotlib sklearn and keras**. Since all modules are included in **anaconda3**, users could resolve module dependencies by installing anaconda3 (2 is not suggested) on their official website https://www.anaconda.com/. Alternatively, the users can install the module manually, for example using pip or another installer. If using pip to install the dependent modules, the command is:

```pip install numpy```

 or
 
```pip install numpy --user```

## 2.2 Set Search Path
After installing the dependent modules, if the working path is the root directory of the extracted folder (that is, the folder where the manual is located), users can use autoBioSeqpy directly in the command line window (CMD window).  
If users want to use it in their own python script, there are two ways to add modules to the search path:
1)	If autoBioSeqpy is already in the python search path, adding a line to the python script is sufficient:

```import autoBioSeqpy```

or

```from autoBioSeqpy import *```

2)	Otherwise, users could add the location into sys.path:

```
import sys
libPath = /the/path/of/the/folder
sys.path.append(libPath)
```

Then all modules are available. An example in jupyter notebook is provided, which uses the provided module for data processing and users can get it in 'notebook/tutorial in jupyter notebook.html'.
# 3 Quick Start
There are two ways to use autoBioSeqpy, one is to use the script running.py as a standalone application, and the other is to integrate it into a python script as a module. We will introduce both ways in separated sections.
## 3.1 Using autoBioSeqpy as Standalone Application
### 3.1.1 Training and predict
If the dependent modules are installed (in section 2.1), a standalone script running.py is available. To test it, just open a command line window (or terminal in Linux) and make the working path (i.e. current folder) to the location of autoBioSeqpy. Then test:

```python running.py --help```

if the help document is showed without error, it’s available. Users can then perform a shot test: 

```python running.py --dataType protein --dataEncodingType dict  --dataTrainFilePaths examples/typeIII_secreted_effectors_prediction/data/train_pos.txt examples/typeIII_secreted_effectors_prediction/data/train_neg.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/typeIII_secreted_effectors_prediction/data/test_pos.txt examples/typeIII_secreted_effectors_prediction/data/test_neg.txt --dataTestLabel 1 0 --modelLoadFile examples/typeIII_secreted_effectors_prediction/model/protein_CNN_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 60 --epochs 20 --spcLen 100 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt ```

The use of parameters is in “parameters” section or in the help document generated by ‘--help’.
Since there are too many parameters to write on the command line, an alternative way is to write the parameters to a text file, for example the file parameters.txt contains the details information (All spaces below can be changed to line breaks):

```
--dataType protein 
--dataEncodingType dict  
--dataTrainFilePaths examples/typeIII_secreted_effectors_prediction/data/train_pos.txt examples/typeIII_secreted_effectors_prediction/data/train_neg.txt 
--dataTrainLabel 1 0 
--dataTestFilePaths examples/typeIII_secreted_effectors_prediction/data/test_pos.txt examples/typeIII_secreted_effectors_prediction/data/test_neg.txt 
--dataTestLabel 1 0 
--modelLoadFile examples/typeIII_secreted_effectors_prediction/model/protein_CNN_model.py 
--verbose 1 
--outSaveFolderPath tmpOut 
--savePrediction 1 
--saveFig 1 
--batch_size 60 
--epochs 20 
--spcLen 100 
--shuffleDataTrain 1 
--modelSaveName tmpMod.json 
--weightSaveName tmpWeight.bin 
--noGPU 0 
--paraSaveName parameters.txt
```

This file can then be used in a command line:

`python running.py --paraFile parameters.txt`

### 3.1.2 Predict using the built model
Sometimes users will want to use the built model to predict the new data, and predicting.py is available. Since the data encoding during training depends on the parameters, few parameters are required during training. Using the same example as in section 3.1.1, the command line becomes:

```python running.py --dataType protein --dataEncodingType dict  --dataTrainFilePaths examples/typeIII_secreted_effectors_prediction/data/train_pos.txt examples/typeIII_secreted_effectors_prediction/data/train_neg.txt --dataTrainLabel 1 0 --dataTestFilePaths examples/typeIII_secreted_effectors_prediction/data/test_pos.txt examples/typeIII_secreted_effectors_prediction/data/test_neg.txt --dataTestLabel 1 0 --modelLoadFile examples/typeIII_secreted_effectors_prediction/model/protein_biLSTM_model.py --verbose 1 --outSaveFolderPath tmpOut --savePrediction 1 --saveFig 1 --batch_size 60 --epochs 20 --spcLen 100 --shuffleDataTrain 1 --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt ```

The three parameters highlighted above are necessary for making predictions, where **--modelSaveName** and **--weightSaveName** are the keras model files and related weights, and **--paraSaveName** are the parameters used when training. Then, for prediction, the command line will become:

```python predicting.py --paraFile tmpOut/parameters.txt --dataTestFilePaths examples/typeIII_secreted_effectors_prediction/data/test_pos.txt --predictionSavePath tmpout/indPredictions.txt```

That is, if the test data and parameters are sufficient (because the module and weight are recorded in parameters.txt), if the user wants to print the output to STDOUT, you can ignore **--predictionSavePath**.
3.2 Using autoBioSeqpy in Other Work
Because autoBioSeqpy can encode FASTA sequences into matrices, sometimes users may just want to use feature encoding instead of modeling. The autoBioSeqpy can be used as a module, so it can be used for other tasks. We provided a jupyter notebook to explain how to use it, so please open the file in **notebook/ tutorial in jupyter notebook.ipynb** in jupyter notebook. If the jupyter notebook is not installed, users could use the HTML and PDF version alternatively (but only for reading, no interaction included in pure HTML and PDF files).

# 4. Conclusion
This document is provided for users to know autoBioSeqpy. As an open source tool, we have documented all the code and function, but this document is still a better way to understand the framework. For more details, please see the word file 'manual.docx'. 
We looking forward to receiving any bug reports and suggestions, please feel free to contact us anytime (ljs@swmu.edu.cn)

# Citing autoBioSeqpy
autoBioSeqpy is published at Journal of Chemical Information and Modeling. The link is https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00409 and the related BibTeX is provided below:

@article{jing2020autobioseqpy,
  title={autoBioSeqpy: a deep learning tool for the classification of biological sequences},
  author={Jing, Runyu and Li, Yizhou and Xue, Li and Liu, Fengjuan and Li, Menglong and Luo, Jiesi},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  number={8},
  pages={3755--3764},
  year={2020},
  publisher={ACS Publications}
}

Please use is as needed.