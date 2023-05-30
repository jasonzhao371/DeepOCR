import argparse
import re
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
def change_char_label(seq,pn,flag=1):
    bed_dict_pos = {}
    out = open(pn+".txt", 'w')
    for line in open(seq, 'r'):
        if line[0] == ">":
            se = line.strip()
            bed_dict_pos[se] = []
        else:
            bed_dict_pos[se].append(line.strip())
    for keys, val in bed_dict_pos.items():
        bed_dict_pos[keys] = ''.join(val)
        out.write(str(flag) + '\t' + bed_dict_pos[keys] + '\n')
    out.close()


def one_hot(filename,nuc_number=1000):#nuc_number:every_sequence_final_nucleotide_number
    f = open(filename, 'r')
    sequences = f.readlines()
    num = len(sequences)
    data = np.empty((num, nuc_number, 4), dtype='int')
    label = np.empty((num,), dtype="int")
    for i in range(num):
        line = sequences[i].replace('\n', '')
        list_line = re.split('\s+', line)
        one_sequence = list_line[1]
        #one_sequence = list_line
        #print(one_sequence)
        for j in range(nuc_number):
            if j <= len(one_sequence) - 1:
                if re.findall(one_sequence[j], 'A|a'):
                    data[i, j, :] = np.array([1, 0, 0, 0], dtype='int32')
                if re.findall(one_sequence[j], 'C|c'):
                    data[i, j, :] = np.array([0, 1, 0, 0], dtype='int32')
                if re.findall(one_sequence[j], 'G|g'):
                    data[i, j, :] = np.array([0, 0, 1, 0], dtype='int32')
                if re.findall(one_sequence[j], 'T|t'):
                    data[i, j, :] = np.array([0, 0, 0, 1], dtype='int32')
                if re.findall(one_sequence[j], 'N|n'):
                    data[i, j, :] = np.array([0, 0, 0, 0], dtype='int32')
            else:
                data[i, j, :] = np.array([0, 0, 0, 0], dtype='int32')
        label[i] = list_line[0]
    return data, label

def main():
    parser = argparse.ArgumentParser(description='DNA Sequence Processing')
    parser.add_argument('--out', type=str, default='./', help='Output directory')
    parser.add_argument('--pos', type=str, default='pos.fa', help='Positive samples file')
    parser.add_argument('--neg', type=str, default='neg.fa', help='Negative samples file')
    parser.add_argument('--random', type=float, help='Random_split',required=False)
    parser.add_argument('--fold', type=int, help='cross validation',required=False)
    args = parser.parse_args()
    pos=args.pos
    neg=args.neg
    out=args.out
    random=args.random
    fold=args.fold
    #onehot label
    change_char_label(pos,pn=out+"pos_1hot",flag=1)
    change_char_label(neg, pn=out + "neg_1hot",flag=0)
    sys.stdout.flush()
    #onehot label
    data_pos_1,label_pos_1=one_hot(out+"pos_1hot.txt")
    data_neg_1, label_neg_1 = one_hot(out + "neg_1hot.txt")
    sys.stdout.flush()
    os.remove(out+"pos_1hot.txt")
    os.remove(out+"neg_1hot.txt")
    #onehot
    seq = np.concatenate((data_pos_1, data_neg_1), axis=0)
    tar = np.concatenate((label_pos_1, label_neg_1), axis=0)
    np.random.seed(12345)
    indices = np.arange(seq.shape[0])
    np.random.seed(12345)
    np.random.shuffle(indices)
    seq= seq[indices]
    Y = tar[indices]
    np.random.seed(12345)
    np.save(out+'data_onehot.npy', seq)
    np.save(out+'label.npy', Y)
    if args.random:
       hot1_train, hot1_test,y_train,y_test = train_test_split(seq,Y,test_size=random,random_state=123)
       np.save(dir+'data_train_onehot.npy', hot1_train)
       np.save(dir+'data_test_onehot.npy', hot1_test)
       np.save(dir+'label_train.npy', y_train)
       np.save(dir+'label_test.npy', y_test)
    if args.fold:
       skf=StratifiedKFold(n_splits=10,random_state=fold,shuffle=True)
       i=0
       
    
       
    
    

if __name__ == '__main__':
    main()
    

      
      
      
      
      
      

