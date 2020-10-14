# metrics for predictions on the test set of dr
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

ds_dr_test = np.load('data_processed/dr_test.npy') # ground truth
ds_dr_deepm6aseq_pos = open("data_processed/deepm6aseq_predictions/dr_pos_test", "r")
ds_dr_deepm6aseq_neg = open("data_processed/deepm6aseq_predictions/dr_neg_test", "r")

count = 0
y_pred = []
y_true = []
tn = 0
tp = 0
fn = 0
fp = 0
for line in ds_dr_deepm6aseq_pos:
	bool_val = line.split('(')[1][0]
	# if bool_val == '+':
	# 	location = int(line.split(';')[4].split('::')[0])
	# else:
	# 	location = -1
	location = int(line.split(';')[4].split('::')[0])
	m6alocation = int(ds_dr_test[count][1])

	for j in range(101):
		if ds_dr_test[count][0][j] == 'A':
			print(ds_dr_test[count][0][j], m6alocation, location, j)
			if j == m6alocation:
				if j == location:
					print("Case 1")
					y_true.append(1)
					y_pred.append(1)
				else:
					print("Case 2")
					y_true.append(1)
					y_pred.append(0)
			else:
				if j == location:
					print("Case 3")
					y_true.append(0)
					y_pred.append(1)
				else:
					print("Case 4")
					y_true.append(0)
					y_pred.append(0)
	count += 1

print(count)

for line in ds_dr_deepm6aseq_neg:
	bool_val = line.split('(')[1][0]
	# if bool_val == '+':
	# 	location = int(line.split('::')[1][0])
	# else:
	# 	location = -1
	location = int(line.split('::')[1][0])
	m6alocation = int(ds_dr_test[count][1])

	for j in range(101):
		if ds_dr_test[count][0][j] == 'A':
			print(ds_dr_test[count][0][j], m6alocation, location, j)
			if j == m6alocation:
				if j == location:
					print("Case 1")
					y_true.append(1)
					y_pred.append(1)
				else:
					print("Case 2")
					y_true.append(1)
					y_pred.append(0)
			else:
				if j == location:
					print("Case 3")
					y_true.append(0)
					y_pred.append(1)
				else:
					print("Case 4")
					y_true.append(0)
					y_pred.append(0)
	count += 1

print(count, len(ds_dr_test))
cm = confusion_matrix(y_true, y_pred)
print(cm, np.sum(cm))
print(classification_report(y_true, y_pred))

'''
dr Test: DeepM6ASeq:
[[167716    830]
 [     0   2826]] 171372
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    168546
           1       0.77      1.00      0.87      2826

    accuracy                           1.00    171372
   macro avg       0.89      1.00      0.93    171372
weighted avg       1.00      1.00      1.00    171372
'''
