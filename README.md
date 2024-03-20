# DeepOCR:a multi-species deep learning-based framework for accurate identification of open chromatin regions in livestock


## INSTALLATION 
git clone https://github.com/jasonzhao371/DeepOCR.git

## Requirements
- Scikit-learn (http://scikit-learn.org/)
- tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/)
- bedtools (https://bedtools.readthedocs.io/en/latest/)
- h5py (pip install h5py)

# Instructions

## Step 1.Preprocessing
### To preprocess the sequences, use the following command:
```shell
python ./src/preprocess.py --out <output directory> --pos <positive samples file> --neg <negative samples file>
```
### It will output two files including the sequence encoding file and the label encoding file in the output directory.
#### Arguments
  
- output directory: the output file path of the final processed data(npy format)
  
- positive samples file: positive samples(.fa)
  
 ```
 e.g.
 >Chr:start-end
  AAAGTTTATTTGAGGCTGGAACAGCACCAAGGGTATAAATGGAAAAAACAGAAGT...
 ```
negative samples file: negative samples(.fa)
 ```
 e.g.
 >Chr:start-end
  AGGTGTTAACTTTTAAAGAAGAATATATTAAGTTATGCCTACCGTGGAATAAGGT...
 ```
#### Notes
If your input file is in bed format,first you need to extract the fasta sequence using bedtools.
```
$ bedtools getfasta -fi <Reference Genome> -bed <your bed format file> -s -fo <Output file>
```
## Step 2.Training and evaluation
### To train and evaluate DeepOCR, use the following command:
```shell
python ./src/train.py --out <output directory> --seq <sequence encoding file> --label <label encoding file> --val 0.1 <--random 0.2 > <--fold 10 >
```
### It will output the trained model in the output directory.
#### Arguments

- output directory: the output path of the model 
  
- sequence encoding file: one of the preprocessed output files
  
- label encoding file: one of the preprocessed output files
  
- val: the proportion of the validation
  
- random: proportion of test sets in random splitting(optional)
  
- fold: number of folds for cross validation( optional)

## Step 3.Prediction
### To predict on DeepOCR, use the following command:
```shell
python ./src/predict.py --out <output directory> --seq <sequence> --model <model file>
```
### It will output the results in the output directory(0: nonOCRs,1:OCRs).
#### Arguments

- output directory: the output path of prediction 

- sequence : the sequences to predict(.fa)

- model file: the trained model(.hdf5)

# Example for users
## If users wish to utilize their own datasets, here is a straightforward example illustrating the entire process from data preparation to complete model training and prediction (all files are located in the 'example' directory).
### 1.Preprocessing
```shell
python ./src/preprocess.py --out ./example/ --pos ./example/pos_test.fa --neg ./example/neg_test.fa
```
#### It will output two files including the sequence encoding file(./example/data_onehot.npy) and the label encoding file(./example/label.npy)

### 2.Training and evaluation
```shell
python ./src/train.py --out ./example/ --seq ./example/data_onehot.npy  --label ./example/label.npy --val 0.1 --random 0.2
```
#### It will output the trained model(./example/model_random.hdf5)

### 3.Prediction
```shell
python ./src/predict.py --out ./example --seq ./example/test.fa --model ./example/model_random.hdf5
```  
 #### It will output the results(./example/pred.csv)



