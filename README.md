# DeepOCR:a multi-species deep learning-based frame-work for accurate identification of open chromatin regions in livestock

# INSTALLATION 
git clone https://github.com/jasonzhao371/DeepOCR/

# Requirements
- Scikit-learn(http://scikit-learn.org/)
- tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/)
- bedtools
- h5py

# Preprocessing

```shell
python ./src/preprocess.py --out <Output directory> --pos <Positive samples file> --neg <Negative samples file>
```


## Arguments:
  
  Output directory: the output file path of the final processed data(npy format)
  
  
  Positive samples file: positive samples(.fa)
  
  ```
  e.g.
  >Chr:start-end
   AAAGTTTATTTGAGGCTGGAACAGCACCAAGGGTATAAATGGAAAAAACAGAAGT...
  ```
  Negative samples file: negative samples(.fa)
  ```
  e.g.
  >Chr:start-end
   AGGTGTTAACTTTTAAAGAAGAATATATTAAGTTATGCCTACCGTGGAATAAGGT...
  ```
 

## Notes:
IF your input file is in bed format,first you need to extract the fasta sequence using bedtools.
```
$ bedtools getfasta -fi <Reference Genome> -bed <your bed format file> -s -fo <Output file>
```
It will output two files including the sequence encoding file and the label encoding file

# Training and evaluation
  
```shell
python ./src/train.py --out <Output directory> --seq <sequence encoding file> --label <label encoding file> --val 0.1 <--random 0.1> <--fold 10>
```
## Arguments:
  ```
  Output directory: the output path of the model 
  
  sequence encoding file: one of the preprocessed output files
  
  label encoding file: one of the preprocessed output files
  
  --val: the proportion of the validation
  
  --random: proportion of test sets in random splitting
  
  --fold: number of folds for cross validation
  ```
  
  
 
