
def compute_tp_tn_fn_fp(y_test, predictions):
	'''
	True positive - actual = 1, predicted = 1
	False positive - actual = 1, predicted = 0
	False negative - actual = 0, predicted = 1
	True negative - actual = 0, predicted = 0
	'''
	tp = sum((y_test == 1) & (predictions == 1))
	tn = sum((y_test == 0) & (predictions == 0))
	fn = sum((y_test == 1) & (predictions == 0))
	fp = sum((y_test == 0) & (predictions == 1))
	return tp, tn, fp, fn


def compute_accuracy(tp, tn, fn, fp):
	'''
	Accuracy = TP + TN / FP + FN + TP + TN
	'''
	return ((tp + tn) * 100)/ float( tp + tn + fn + fp)


def compute_precision(tp, fp):
	'''
	Precision = TP  / FP + TP 
	'''
	return (tp  * 100)/ float( tp + fp)


def compute_recall(tp, fn):
	'''
	Recall = TP /FN + TP 
	'''
	return (tp  * 100)/ float( tp + fn)


def compute_f1_score(y_true, y_pred):
    # calculates the F1 score
    tp, tn, fp, fn = compute_tp_tn_fn_fp(y_true, y_pred)
    precision = compute_precision(tp, fp)/100
    recall = compute_recall(tp, fn)/100
    f1_score = (2*precision*recall)/ (precision + recall)
    return f1_score*100