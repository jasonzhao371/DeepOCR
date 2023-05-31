# DeepOCR:a multi-species deep learning-based frame-work for accurate identification of open chromatin regions in livestock

# INSTALLATION 
git clone https://github.com/jasonzhao371/DeepOCR/

# Requirements
- Scikit-learn(http://scikit-learn.org/)
- tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/)




# Preprocessing


python ./src/preprocess.py --out <Output directory> --pos <Positive samples file(.fa)> --neg <Negative samples file(.fa)>
  

----------------Training and evaluation---------------------------------
  
 
python ./src/train.py --out <Output directory> --seq data_onehot.npy --label label.npy --val 0.1 <--random 0.1> <--fold 10>
