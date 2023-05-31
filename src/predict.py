import argparse
import re
import numpy as np
import sys
import os
from mode_DeepOCR import mode_DeepOCR
import h5py

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, \
    BatchNormalization, Activation, Flatten, Embedding, ReLU,GlobalAveragePooling1D, Input, concatenate
from tensorflow.keras import backend as K
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
def change_char_label(seq,pn):
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
        out.write(bed_dict_pos[keys] + '\n')
    out.close()
    
def onehot(filename,nuc_number=1000):
    f = open(filename, 'r')
    sequences = f.readlines()
    num = len(sequences)
    data = np.empty((num, nuc_number, 4), dtype='int')
    label = np.empty((num,), dtype="int")
    for i in range(num):
        line = sequences[i].replace('\n', '')
        list_line = re.split('\s+', line)
        one_sequence = list_line
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
    return data
   
def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('--out', type=str,help='Output directory')
    parser.add_argument('--model', type=str,help='Model file')
    parser.add_argument('--seq', type=str, help='Positive samples file')
    args = parser.parse_args()
    out=args.out
    modelf=args.model
    seq=args.seq
    change_char_label(seq,pn=out+"pos_1hot")
    sys.stdout.flush()
    data=onehot(out+"pos_1hot.txt")
    sys.stdout.flush()
    os.remove(out+"pos_1hot.txt")
    loss = tf.keras.losses.binary_crossentropy
    model_b = mode_DeepOCR()
    model_b.load_weights(modelf)
    model_b.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    pred_rate = model_b.predict(data)
    pred = np.where(pred_rate > 0.5, 1, 0)
    pred=pd.DataFrame(pred)
    pred.to_csv(out+"pred.csv",index=False,header=None,sep=",")
    
   
    
    
    
    
    
if __name__ == '__main__':
    main()
    

      
      
      
      
      
      

