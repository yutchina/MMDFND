# -*-codeing = utf-8 -*-
# @Time : 2023-12-121:07
# @Author : 童宇
# @File : utils.py
# @software :
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
def clipdata2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda(),
        'clip_image':batch[5].cuda(),
        'clip_text': batch[6].cuda()
    }
    return batch_data
def data2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda()
    }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

"""
def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except Exception as e:
            metrics_by_category[c] = {
                'auc': 0
            }

        metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
        y_pred = np.around(np.array(y_pred)).astype(int)
        metrics_by_category['metric'] = f1_score(y_true, y_pred, average='macro')
        metrics_by_category['recall'] = recall_score(y_true, y_pred, average='macro')
        metrics_by_category['precision'] = precision_score(y_true, y_pred, average='macro')
        metrics_by_category['acc'] = accuracy_score(y_true, y_pred)

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'precision': precision_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                             average='macro').round(4).tolist(),
                'recall': recall_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                       average='macro').round(4).tolist(),
                'fscore': f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                   average='macro').round(4).tolist(),
                'auc': metrics_by_category[c]['auc'],
                'acc': accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)).round(4)
            }
        except Exception as e:
            metrics_by_category[c] = {
                'precision': 0,
                'recall': 0,
                'fscore': 0,
                'auc': 0,
                'acc': 0
            }
    return metrics_by_category
"""

def metricsTrueFalse(y_true, y_pred, category, category_dict):
    y_GT = y_true
    metricsTrueFalse = metrics(y_true, y_pred, category, category_dict)
    fake = {}
    real = {}
    THRESH = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    realnews_TP, realnews_TN, realnews_FP, realnews_FN = [0]*9, [0]*9, [0]*9, [0]*9
    fakenews_TP, fakenews_TN, fakenews_FP, fakenews_FN = [0]*9, [0]*9, [0]*9, [0]*9
    realnews_sum, fakenews_sum = [0] * 9, [0] * 9
    for thresh_idx, thresh in enumerate(THRESH):
        for i in range(len(y_pred)):
            if y_pred[i]< thresh:y_pred[i]=0
            else:y_pred[i]=1
        for idx in range(len(y_pred)):
            if y_GT[idx] == 1:
                #  FAKE NEWS RESULT
                fakenews_sum[thresh_idx] += 1
                if y_pred[idx] == 0:
                    fakenews_FN[thresh_idx] += 1
                    realnews_FP[thresh_idx] += 1
                else:
                    fakenews_TP[thresh_idx] += 1
                    realnews_TN[thresh_idx] += 1
            else:
                # REAL NEWS RESULT
                realnews_sum[thresh_idx] += 1
                if y_pred[idx] == 1:
                    realnews_FN[thresh_idx] += 1
                    fakenews_FP[thresh_idx] += 1
                else:
                    realnews_TP[thresh_idx] += 1
                    fakenews_TN[thresh_idx] += 1

    val_accuracy, real_accuracy, fake_accuracy, real_precision, fake_precision = [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9
    real_recall, fake_recall, real_F1, fake_F1 = [0] * 9, [0] * 9, [0] * 9, [0] * 9
    for thresh_idx, _ in enumerate(THRESH):
        val_accuracy[thresh_idx] = (realnews_TP[thresh_idx]+realnews_TN[thresh_idx])/(realnews_TP[thresh_idx]+realnews_TN[thresh_idx]+realnews_FP[thresh_idx]+realnews_FN[thresh_idx])
        real_accuracy[thresh_idx] = (realnews_TP[thresh_idx])/realnews_sum[thresh_idx]
        fake_accuracy[thresh_idx] = (fakenews_TP[thresh_idx])/fakenews_sum[thresh_idx]
        real_precision[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FP[thresh_idx]))
        fake_precision[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FP[thresh_idx]))
        real_recall[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FN[thresh_idx]))
        fake_recall[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FN[thresh_idx]))
        real_F1[thresh_idx] = 2*(real_recall[thresh_idx]*real_precision[thresh_idx])/max(1,(real_recall[thresh_idx]+real_precision[thresh_idx]))
        fake_F1[thresh_idx] = 2 * (fake_recall[thresh_idx] * fake_precision[thresh_idx]) / max(1,(fake_recall[thresh_idx] + fake_precision[thresh_idx]))
    fake['precision'] =fake_precision[0]
    fake['recall'] =fake_recall[0]
    fake['F1'] =fake_F1[0]
    real['precision'] =real_precision[0]
    real['recall'] =real_recall[0]
    real['F1'] =real_F1[0]
    metricsTrueFalse['real']=real
    metricsTrueFalse['fake'] = fake
    return metricsTrueFalse

def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except ValueError:
            pass

    try:
        metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        pass
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred)

    for c, res in res_by_category.items():
        # precision, recall, fscore, support = precision_recall_fscore_support(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), zero_division=0)
        metrics_by_category[c] = {
            'precision': precision_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                         average='macro').round(4).tolist(),
            'recall': recall_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                   average='macro').round(4).tolist(),
            'fscore': f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(
                4).tolist(),

            #'auc': metrics_by_category[c]['auc'],
            'acc': accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)).round(4)
        }
    return metrics_by_category
class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)