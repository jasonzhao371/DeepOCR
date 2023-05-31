# DeepOCR:a multi-species deep learning-based frame-work for accurate identification of open chromatin regions in livestock

## INSTALLATION 
git clone https://github.com/jasonzhao371/DeepOCR/

## Requirements
- Scikit-learn(http://scikit-learn.org/)
- tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/)
- bedtools
- h5py

## Preprocessing

```shell
python ./src/preprocess.py --out <output directory> --pos <positive samples file> --neg <negative samples file>
```


### Arguments
  
output directory: the output file path of the final processed data(npy format)
  
positive samples file: positive samples(.fa)
  
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
It will output two files including the sequence encoding file and the label encoding file in the output directory 
### Notes
IF your input file is in bed format,first you need to extract the fasta sequence using bedtools.
```
$ bedtools getfasta -fi <Reference Genome> -bed <your bed format file> -s -fo <Output file>
```

## Training and evaluation
  
```shell
python ./src/train.py --out <output directory> --seq <sequence encoding file> --label <label encoding file> --val 0.1 <--random 0.1> <--fold 10>
```
### Arguments

output directory: the output path of the model 
  
sequence encoding file: one of the preprocessed output files
  
label encoding file: one of the preprocessed output files
  
--val: the proportion of the validation
  
--random: proportion of test sets in random splitting(optional)
  
--fold: number of folds for cross validation( optional)

### Notes
It will output the trained model in the output directory

## Prediction
```shell
python ./src/predict.py --out <output directory> --seq <sequence> --model <model file>
```  
### Arguments

output directory: the output path of prediction 
  
sequence : the sequence to predict(.fa)

model file: the trained model(.hdf5)

### Notes
It will output the results in the output directory(0: nonOCRs,1:OCRs)
