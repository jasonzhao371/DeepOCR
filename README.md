DeepOCR: a multi-species deep learning-based frame-work for accurate identification of open chro-matin regions in livestock


INSTALLATION To run DeepOCR: First install scikit-learn (http://scikit-learn.org/), tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/)



----------------Preprocessing---------------------------------


python ./src/preprocess.py --out <Output directory> --pos <Positive samples file(.fa)> --neg <Negative samples file(.fa)>
  

----------------Training and evaluation---------------------------------
  
 
python ./src/train.py --out <Output directory> --seq data_onehot.npy --label label.npy --val 0.1 <--random 0.1> <--fold 10>
