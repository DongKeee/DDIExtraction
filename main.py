# -*- coding:UTF-8 -*-
#词嵌入：当前词，位置12；模型：DilatedCNN
from classEntity import Document
from Datasets import *
from dataProcess import load_data,Initial
import numpy as np
def focal_loss(classes_num, gamma=2.0, alpha=0.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
                prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
                target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = tf.zeros_like(prediction_tensor)
        one_minus_p = tf.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        # 2# get balanced weight alpha
        classes_weight = tf.zeros_like(prediction_tensor)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = tf.where(target_tensor > zeros, classes_weight, zeros)

        # 3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(
            K.ones_like(prediction_tensor) / nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def getPositionVec(dis):
    if dis <= -31:
        return 0
    elif -30 <= dis <= -21:
        return 1
    elif -20 <= dis <= -11:
        return 2
    elif -10 <= dis <= -6:
        return 3
    elif dis == -5:
        return 4
    elif dis == -4:
        return 5
    elif dis == -3:
        return 6
    elif dis == -2:
        return 7
    elif dis == -1:
        return 8
    elif dis == 0:
        return 9
    elif dis == 1:
        return 10
    elif dis == 2:
        return 11
    elif dis == 3:
        return 12
    elif dis == 4:
        return 13
    elif dis == 5:
        return 14
    elif 6 <= dis <= 10:
        return 15
    elif 11 <= dis <= 20:
        return 16
    elif 21 <= dis <= 30:
        return 17
    elif 31 <= dis:
        return 18


if __name__=="__main__":
    #load data
    train_path = "data/train/"
    test_path = "data/test/"
    train_ = Datasets(filename=train_path)
    train_data = train_.features
    test_ = Datasets(filename=test_path)
    test_data = test_.features
    word_dict = initial.word_dict

    print("get the train data")
    train_array = list()
    train_label = list()
    train_postion1 = list()
    train_postion2 = list()
    train_entityNum = {}
    for feature in train_data:
        if feature['negative'] is False:
            train_label.append(feature['label'])
            train_array.append(feature['all_sequence'])
            if train_entityNum.__contains__(feature['entityNum']):
                train_entityNum[feature['entityNum']]=train_entityNum[feature['entityNum']]+1
            else:
                train_entityNum[feature['entityNum']]=1
            # train_entityNum.append(feature['entityNum'])
            position_vec1 = []
            position_vec2 = []
            for i in range(len(feature['all_sequence'])):
                posEnt1 = feature['e1_pos']
                posEnt2 = feature['e2_pos']
                dis_1 = i - posEnt1
                dis_2 = i - posEnt2
                position_vec1.append(getPositionVec(dis_1))
                position_vec2.append(getPositionVec(dis_2))
            # train_postion1.append(np.array(position_vec1))
            # train_postion2.append(np.array(position_vec2))
            train_postion1.append(position_vec1)
            train_postion2.append(position_vec2)
    # for key in sorted(train_entityNum.keys()):
    #     print(key,':',train_entityNum[key])
    #     # openfile.write(str(key)+":"+str(dic[key])+"\n")
    train_array = np.array(train_array)
    train_postion1 = np.array(train_postion1)
    train_postion2 = np.array(train_postion2)
    print(train_array.shape)

    print("get the test data")
    test_array = list()
    test_label = list()
    test_postion1 = list()
    test_postion2 = list()
    test_entityNum = {}
    for feature in test_data:
        if feature['negative'] is False:
            test_label.append(feature['label'])
            test_array.append(feature['all_sequence'])
            if test_entityNum.__contains__(feature['entityNum']):
                test_entityNum[feature['entityNum']]=test_entityNum[feature['entityNum']]+1
            else:
                test_entityNum[feature['entityNum']]=1
            position_vec1 = []
            position_vec2 = []
            for i in range(len(feature['all_sequence'])):
                posEnt1 = feature['e1_pos']
                posEnt2 = feature['e2_pos']
                dis_1 = i - posEnt1
                dis_2 = i - posEnt2
                position_vec1.append(getPositionVec(dis_1))
                position_vec2.append(getPositionVec(dis_2))
            test_postion1.append(position_vec1)
            test_postion2.append(position_vec2)
    test_array = np.array(test_array)
    test_postion1 = np.array(test_postion1)
    test_postion2 = np.array(test_postion2)

    print(test_array.shape)
    #构建左、右侧文本
    doc_x_train = train_array
    # We shift the document to the right to obtain the left-side contexts.
    left_x_train = np.array([[len(word_dict)] + t_one[:-1].tolist() for t_one in train_array])
    # We shift the document to the left to obtain the right-side contexts.
    right_x_train = np.array([t_one[1:].tolist() + [len(word_dict)] for t_one in train_array])

    doc_x_test = np.array(test_array)
    # We shift the document to the right to obtain the left-side contexts.
    left_x_test = np.array([[len(word_dict)] + t_one[:-1].tolist() for t_one in test_array])
    # We shift the document to the left to obtain the right-side contexts.
    right_x_test = np.array([t_one[1:].tolist() + [len(word_dict)] for t_one in test_array])
    


    from keras.callbacks import ModelCheckpoint
    from keras.layers import Embedding
    import numpy as np
    from keras.layers import Dense, Input, Flatten, merge
    from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Bidirectional, Dropout
    from keras.models import Model
    import keras.backend as K
    from CallBackMy import CallBackMy
    import copy
    from keras.utils import np_utils
    from keras.layers.merge import concatenate
    from keras.layers.normalization import BatchNormalization
    from keras import regularizers
    # embedding layer
    word_embedding = Embedding(input_dim=word_dict.shape[0],
                              output_dim=200,
                              input_length=150,
                              weights=[word_dict],
                              trainable=False)
    position_embedding = Embedding(input_dim=19,
                                   output_dim=15,
                                   input_length=150,
                                   trainable=True)
    print('Training model.')
    input_word = Input(shape=(150,), dtype='int32', name='input_word')
    word_fea = word_embedding(input_word)  # trainable=False

    input_pos1 = Input(shape=(150,), dtype='int32', name='input_pos1')
    pos_fea1 = position_embedding(input_pos1)

    input_pos2 = Input(shape=(150,), dtype='int32', name='input_pos2')
    pos_fea2 = position_embedding(input_pos2)


    left_context = Input(shape=(150,), dtype="int32", name='left_context')
    l_embedding = word_embedding(left_context)

    right_context = Input(shape=(150,), dtype="int32", name='right_context')
    r_embedding = word_embedding(right_context)
    

    window = 3
    filters=64
  
    batchsize=32
    hidden_dim_1=200

    forward = LSTM(hidden_dim_1, return_sequences=True)(l_embedding)
    backward = LSTM(hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)

    sequence_input = concatenate([forward, word_fea, backward], axis=-1)
    sequence_input = Dense(256, activation='relu')(sequence_input)
    sequence_input = Dense(64, activation='relu')(sequence_input)
    sequence_input = concatenate([sequence_input,pos_fea1,pos_fea2], axis=-1)


    x1 = Conv1D(64, 3, activation='relu', dilation_rate=2, padding='same')(sequence_input)
    x1 = MaxPooling1D(5)(x1)
    x1 = Conv1D(64, 5, activation='relu', dilation_rate=2, padding='same')(x1)
    x1 = MaxPooling1D(25)(x1)
    x2 = Conv1D(64, 3, activation='relu', padding='same')(sequence_input)
    x2 = MaxPooling1D(5)(x2)
    x2 = Conv1D(64, 5, activation='relu', padding='same')(x2)
    x2 = MaxPooling1D(25)(x2)
    x = concatenate([x1, x2], axis=-1)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    preds = Dense(5, activation='softmax')(x)
    classes_num=[188,822,1669,1319,19012]
    model = Model(inputs=[left_context, input_word, right_context,input_pos1,input_pos2], outputs=preds)
    model.summary()
    model.compile(
#loss='categorical_crossentropy',
loss=focal_loss(classes_num),
                  optimizer='adam',
                  metrics=['accuracy', f1_score, precision, recall])
    train_label_cat = np_utils.to_categorical(train_label, 5)  # 将类别按照one-hot进行编码
    test_backup = np.array(copy.deepcopy(test_label))
    test_label_cat = np_utils.to_categorical(test_label, 5)
    callbackmy = CallBackMy(test_array=test_array,
                            windows=window,
                            lcontext=left_x_test,
                            rcontext=right_x_test,
                            testpos1=test_postion1,
                            testpos2=test_postion2,
                            test_backup=test_backup,
                            log_dict={"filters": filters, "batchsize": batchsize},
                            filename="log//process//8.txt",
                            filename2="log//result//8.txt")

    model.fit([left_x_train[0:20], train_array[0:20], right_x_train[0:20], train_postion1[0:20], train_postion2][0:20], train_label_cat[0:20],
              batch_size=batchsize,
              epochs=50,
              validation_data=([left_x_test, test_array, right_x_test, test_postion1, test_postion2], test_label_cat),
              callbacks=callbackmy,verbose=1)
    predict = []
    predicted=model.predict([left_x_test, test_array, right_x_test, test_postion1, test_postion2], verbose=0)
    predicted = predicted.tolist()
    for i in range(len(predicted)):
        predict.append(predicted[i].index(max(predicted[i])))
    from MicroCalculate import *
    P, R, F = calculateMicroValue(y_pred=predict, y_true=test_backup, labels=[0, 1, 2, 3, 4],
                                                 filename='111', filename2='222')
    print(P,R,F)