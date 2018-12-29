# -*- coding:UTF-8 -*-
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
def calculateMicroValue(y_pred, y_true, labels=None,filename='',filename2=''):
    if labels is None:
        labels = [0, 1]
    all_label_result = {}
    openfileProcess = open(filename, 'a')
    openfileResult = open(filename2, 'a')
    confusionMetrix=confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    print(confusionMetrix)
    for label in labels:#计算各个类别下的 tp  fp  fn
        all_label_result[label] = calculateF(y_pred, y_true, label)
    # confusionMetrix= calculateMetrix(y_pred, y_true)
    print("Corresponding result -->", all_label_result)
    openfileProcess.write("confusionMetrix[0,1,2,3,4]" + str(confusionMetrix)  + "\n")
    openfileProcess.write("Corresponding result -->tp  fp  fn" + str(all_label_result)  + "\n")

    tp = fp = fn = 0.0

    for label in [0,1,2,3,4]:#计算每类样本的 tp  fp  fn
        tp = all_label_result[label][0]
        fp = all_label_result[label][1]
        fn = all_label_result[label][2]
        p = tp / (tp + fp + 0.00001)
        r = tp / (tp + fn + 0.00001)
        f = 2 * p * r / (p + r + 0.00001)
        print("for label", label, "\t p=", p, "\tr=", r, "\tf=", f)
        openfileProcess.write("for label:"+str(label)+ "\tP:" + str(p) + "\tR:" + str(r) + "\tF:" + str(f) + "\n")

    tp = fp = fn = 0.0
    for label in [0,1,2,3,4]:#计算每类的 p r f
        tp += all_label_result[label][0]
        fp += all_label_result[label][1]
        fn += all_label_result[label][2]
    p = tp / (tp + fp + 0.00001)
    r = tp / (tp + fn + 0.00001)
    f = 2 * p * r / (p + r + 0.00001)
    openfileProcess.write("for all label:\tP:" + str(p) + "\tR:" + str(r) + "\tF:" + str(f) + "\n")
    openfileResult.write("all class:\tP:" + str(p) + "\tR:" + str(r) + "\tF:" + str(f) + "\n")

    tp = fp = fn = 0.0
    for label in [0, 1, 2, 3]:#计算前四类的 p r f
        tp += all_label_result[label][0]
        fp += all_label_result[label][1]
        fn += all_label_result[label][2]
    p = tp / (tp + fp + 0.00001)
    r = tp / (tp + fn + 0.00001)
    f = 2 * p * r / (p + r + 0.00001)
    openfileProcess.write("for four class label:\tP:" + str(p) + "\tR:" + str(r) + "\tF:" + str(f) + "\n")
    openfileResult.write("four class:\tP:" + str(p) + "\tR:" + str(r) + "\tF:" + str(f) + "\n\n")
    print("for four class  label", "\t p=", p, "\tr=", r, "\tf=", f)  # 计算所有样本的 tp  fp  fn

    rep = classification_report(
        y_pred,
        y_true, digits=4)
    openfileProcess.write(rep + "\n")
    openfileProcess.write("=========================================================================\n")
    openfileProcess.close()
    openfileResult.close()
    return p, r, f


def calculateF(y_pred, y_true, label):
    tp = fp = fn = 0.0
    for left, right in zip(y_pred, y_true):
        if label == left and right == label:
            tp += 1
        if left == label and right != label:
            fp += 1
        if left != label and right == label:
            fn += 1
    return [tp, fp, fn]

