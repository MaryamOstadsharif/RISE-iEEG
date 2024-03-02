"""import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

acc_GN = np.load("F:/maryam_sh/General model/General code/results/Singing_Music/under_sampling/"
                   "2023-12-03-00-15-30_star/accuracy/acc_gen_GNCNN_29.npy")

row_labels=[]
for i in range(29):
    print("Test accuracy of fold", str(i), "= ", acc_GN[i, 0],'/', acc_GN[i, 1],'/', acc_GN[i, 2])
    row_labels.append('fold_'+str(i))

print("mean accuracy =", np.mean(acc_GN[:, 2]))


table = PrettyTable()
table.field_names = ["Fold", "Train Accuracy", "Validation Accuracy", "Test Accuracy"]

for i in range(29):
    table.add_row([f"fold_{i}", acc_GN[i, 0], acc_GN[i, 1], acc_GN[i, 2]])

# Print PrettyTable
print(table)

print('end')"""

import numpy as np
from prettytable import PrettyTable

acc_GN = np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
                   "2024-02-28-13-26-10/accuracy/fscore_GNCNN_each_patient_5.npy")

row_labels=[]
for i in range(9):
    print("Test accuracy of patient", str(i), "= ", np.mean(acc_GN,axis=0)[i])
    row_labels.append('patient_'+str(i))

table = PrettyTable()
table.field_names = ["Patient_ID", "Train Accuracy"]

print('end')