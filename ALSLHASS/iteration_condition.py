import numpy as np

before_pre_label = 0


def whether_iteration(predict_label, iter_num):
    global before_pre_label
    if iter_num == 0:
        before_pre_label = predict_label
        return 0
    else:
        current_pre_label = np.array(predict_label)
        equal_prob = format(sum(current_pre_label != before_pre_label) / len(before_pre_label), '.3f')
        before_pre_label = np.array(predict_label)
        return equal_prob
