import math
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt


def naive_bayes():
    pos_train, neg_train, vocab = load_training_set(0.02, 0.02)
    pos_test, neg_test = load_test_set(0.02, 0.02)
    pos_word_counter = count_word_freq(pos_train)
    neg_word_counter = count_word_freq(neg_train)
    pos_cls_prb = len(pos_train) / (len(pos_train) + len(neg_train))
    neg_cls_prb = len(neg_train) / (len(pos_train) + len(neg_train))
    print("Without log: ")
    predictions_without_log = predict_without_log(pos_word_counter, neg_word_counter, pos_test, neg_test, pos_cls_prb, neg_cls_prb,
                                      len(vocab), 1)
    print("With log: ")
    accuracy = predict(pos_word_counter, neg_word_counter, pos_test, neg_test, pos_cls_prb, neg_cls_prb,
                                   len(vocab), 1)


def count_word_freq(dataset):
    wordfreq_dict = defaultdict(int)
    for sublist in dataset:
        for word in sublist:
            wordfreq_dict[word] += 1
    return wordfreq_dict


def calc_prob(test_set, train, all_word_count, class_prb, alpha):
    alpha = alpha
    prb_all_words = 1.0
    for word in test_set:
        prb_word = (train.get(word, 0) + alpha)/((sum(train.values())) + (alpha * all_word_count))
        prb_all_words *= prb_word
    final_prb = prb_all_words * class_prb
    return final_prb


def calc_log_prob(test_set, train, all_word_count, class_prb, alpha):
    alpha = alpha
    prb_all_words = 0.0
    for word in test_set:
        prb_word = (train.get(word, 0) + alpha)/((sum(train.values())) + (alpha * all_word_count))
        prb_all_words += math.log(prb_word)
    final_prb = prb_all_words + math.log(class_prb)
    return final_prb


def predict(pos_word_counter, neg_word_counter, pos_test, neg_test, pos_cls_prb, neg_cls_prb, vocab_len, alpha):
    tn, tp, fn, fp = 0, 0, 0, 0
    true_data = []
    predictions = []
    for pos_data in pos_test:
        pos_prb = calc_log_prob(pos_data, pos_word_counter, vocab_len, pos_cls_prb, alpha)
        neg_prb = calc_log_prob(pos_data, neg_word_counter, vocab_len, neg_cls_prb, alpha)
        if pos_prb > neg_prb:
            predictions.append(1)
            true_data.append(1)
        else:
            predictions.append(0)
            true_data.append(1)

    for neg_data in neg_test:
        pos_prb = calc_log_prob(neg_data, pos_word_counter, vocab_len, pos_cls_prb, alpha)
        neg_prb = calc_log_prob(neg_data, neg_word_counter, vocab_len, neg_cls_prb, alpha)
        if pos_prb>neg_prb:
            predictions.append(1)
            true_data.append(0)
        else:
            predictions.append(0)
            true_data.append(0)
    accuracy = confusion_matrix(true_data, predictions)
    return accuracy


def predict_without_log(pos_word_counter, neg_word_counter, pos_test, neg_test, pos_cls_prb, neg_cls_prb, vocab_len, alpha):
    tn, tp, fn, fp = 0, 0, 0, 0
    true_data = []
    predictions = []
    for pos_data in pos_test:
        pos_prb = calc_prob(pos_data, pos_word_counter, vocab_len, pos_cls_prb, alpha)
        neg_prb = calc_prob(pos_data, neg_word_counter, vocab_len, neg_cls_prb, alpha)
        if pos_prb > neg_prb:
            predictions.append(1)
            true_data.append(1)
        else:
            predictions.append(0)
            true_data.append(1)

    for neg_data in neg_test:
        pos_prb = calc_prob(neg_data, pos_word_counter, vocab_len, pos_cls_prb, alpha)
        neg_prb = calc_prob(neg_data, neg_word_counter, vocab_len, neg_cls_prb, alpha)
        if pos_prb>neg_prb:
            predictions.append(1)
            true_data.append(0)
        else:
            predictions.append(0)
            true_data.append(0)
    confusion_matrix(true_data, predictions)
    return predictions


def confusion_matrix(true_data, predictions):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and true_data[i] == 1:
            tp += 1
        elif predictions[i] == 0 and true_data[i] == 0:
            tn += 1
        elif predictions[i] == 1 and true_data[i] == 0:
            fp += 1
        elif predictions[i] == 0 and true_data[i] == 1:
            fn += 1
    accuracy = evaluate_model(tp, tn, fp, fn)
    return accuracy


def evaluate_model(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    matrix = [[tp, fn], [fp, tn]]
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Confusion matrix: ", matrix)
    return accuracy


if __name__ == "__main__":
    naive_bayes()
