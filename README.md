# DeepOCR:a multi-species deep learning-based frame-work for accurate identification of open chromatin regions in livestock

# INSTALLATION 
git clone https://github.com/jasonzhao371/DeepOCR/

# Requirements
- Scikit-learn(http://scikit-learn.org/)
- tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/)
- bedtools




# Preprocessing

```shell
python ./src/preprocess.py --out <Output directory> --pos <Positive samples file> --neg <Negative samples file>
```


## Arguments:
  Output directory:The output file path of the final processed data(npy format)
  
  Positive samples file:positive samples(.fa)
  
  Negative samples file:negative samples(.fa)


### Notes:
IF your input file is in bed format,first you need to extract the fasta sequence using the tool bedtools.
```
$ bedtools getfasta -fi <Reference Genome> -bed <your bed format file> -s -fo <Output file>
```
# Training and evaluation
  
```shell
python ./src/train.py --out <Output directory> --seq data_onehot.npy --label label.npy --val 0.1 <--random 0.1> <--fold 10>
```
```
arguments:
  Output directory:
