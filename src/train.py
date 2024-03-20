import h5py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import math
import argparse
plt.switch_backend('agg')
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, \
    BatchNormalization, Activation, Flatten, Embedding, ReLU
from tensorflow.keras.layers import GlobalAveragePooling1D, Input, concatenate, ConvLSTM2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
import os
from tensorflow.keras.metrics import Recall, Precision, AUC, BinaryAccuracy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.backend import squeeze
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MaxPooling2D, Convolution1D, Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    cohen_kappa_score, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    cohen_kappa_score, precision_recall_curve, average_precision_score, roc_auc_score
from mode_DeepOCR import mode_DeepOCR
import tensorflow as tf
from sklearn.model_selection import train_test_split
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random
from sklearn.model_selection import KFold,StratifiedKFold
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
def trainDeepOCR(out, train_seq, train_label, test_seq, test_label,val_split,i):
    train_model = mode_DeepOCR()
    loss = tf.keras.losses.binary_crossentropy
    train_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    train_model.summary()
    checkpoint = ModelCheckpoint(filepath=out + 'model'+str(i)+'.hdf5',
                                 # monitor='val_accuracy',
                                 save_best_only=True,
                                 verbose=1
                                 # save_weights_only=True,
                                 # mode='max'
                                 )

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=20,
                                   verbose=1
                                   # mode='max'
                                   )

    callback_lists = [checkpoint, early_stopping]
    hist = train_model.fit(train_seq, train_label,
                     # batch_size=128,
                     batch_size=64,#64
                     epochs=300,
                     verbose=2,
                     callbacks=callback_lists,
                     validation_split=val_split,
                     shuffle=True
                     # validation_data=(vs, vt)
                     )
    model_b = mode_DeepOCR()
    model_b.load_weights(out + 'model'+str(i)+'.hdf5')
    model_b.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    score = model_b.evaluate(test_seq, test_label, verbose=0)
    acc = score[1]
    pred_rate = model_b.predict(test_seq)
    fpr1, tpr1, thresholds = metrics.roc_curve(test_label, pred_rate)
    auc = metrics.auc(fpr1, tpr1)
    pred = np.where(pred_rate > 0.5, 1, 0)
    f1 = f1_score(test_label, pred)
    cm = confusion_matrix(test_label, pred)
    tp = cm[0, 0]
    fp = cm[1, 0]
    tn = cm[1, 1]
    fn = cm[0, 1]
    MCC = float((tp * tn - fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (
                tp + fn) * (tn + fp) * (tn + fn) != 0 else 'NA'
    total1 = sum(sum(cm))
    Se = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    Sp = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    return acc,auc,Se,Sp,MCC,f1


                        


def main():
  parser = argparse.ArgumentParser(description='Training and evaluation')
  parser.add_argument("--out", required=True, help="path to output dir")
  parser.add_argument("--seq", required=True, help="encoded_sequence")
  parser.add_argument("--label", required=True, help="label of sequence(0/1)")
  parser.add_argument("--val", type=float,help="val_split", required=True)
  parser.add_argument("--random", type=float,help="random_split", required=False)
  parser.add_argument('--fold', type=int, help='cross_validation',required=False)
  args = parser.parse_args()
  out=args.out
  seq=args.seq
  label=args.label
  val=args.val
  random=args.random
  fold=args.fold
  seq = np.load(seq)
  seq = np.array(seq, dtype='float32')
  label = np.load(label)
  label = label.reshape(label.shape[0], 1)
  if args.random:
     i="_random"
     train_seq,test_seq,train_label,test_label = train_test_split(seq,label,test_size=random)
     acc,auc,Se,Sp,MCC,f1=trainDeepOCR(out, train_seq, train_label, test_seq, test_label,val,i)
     print("Acc:",acc)
     print("F1-score:",f1)
     print("Auc:",auc)
     print("Se:",Se)
     print("Sp:",Sp)
     print("MCC:",MCC)
     
  elif args.fold:
     avg_acc=0
     avg_auc=0
     avg_se=0
     avg_sp=0
     avg_mcc=0
     avg_f1=0
     skf=StratifiedKFold(n_splits=fold,shuffle=True)
     i=0
     for train_index, test_index in skf.split(X,y):
       i+=1
       train_seq, test_seq = seq[train_index], seq[test_index]
       train_label,test_label = label[train_index], label[test_index]
       acc,auc,Se,Sp,MCC,f1=trainDeepOCR(out, train_seq, train_label, test_seq, test_label,val,i)
       avg_acc+=acc
       avg_auc+=auc
       avg_se+=Se  
       avg_sp+=Sp
       avg_mcc+=MCC
       avg_f1+=f1
     print("Acc:",avg_acc/fold)
     print("F1-score:",avg_f1/fold)
     print("Auc:",avg_auc/fold)
     print("Se:",avg_se/fold)
     print("Sp:",avg_sp/fold)
     print("MCC:",avg_mcc/fold)
  else :
     print("Must choose --random or --fold")
         
if __name__ == '__main__':
  main()
    
  
