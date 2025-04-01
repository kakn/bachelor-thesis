seconds = 30762.060083389282
hours = seconds / 3600
import numpy as np
hours = np.round(hours, 3)

import matplotlib.pyplot as plt

nums = open('bigcomedy_model_history_log.csv').read()
epochs = []
accs = []
loss = []
val_acc = []
val_loss = []
for jig in nums.split()[1:]:
    pig = jig.split(',')
    epochs.append(float(pig[0]))
    accs.append(float(pig[1]))
    loss.append(float(pig[2]))
    val_acc.append(float(pig[3]))
    val_loss.append(float(pig[4]))

nums2 = open('thriller_model_history_log.csv').read()
epochs2 = []
accs2 = []
loss2 = []
val_acc2 = []
val_loss2 = []
for jig in nums2.split()[1:]:
    pig = jig.split(',')
    epochs2.append(float(pig[0]))
    accs2.append(float(pig[1]))
    loss2.append(float(pig[2]))
    val_acc2.append(float(pig[3]))
    val_loss2.append(float(pig[4]))

#print(accs)
# plt.plot(epochs2, loss2)
# plt.plot(epochs2, val_loss2)
plt.plot(epochs, val_loss)
plt.plot(epochs2, val_loss2)
plt.title("Large Thriller and Comedy validation losses")
#plt.title('Training accuracy of large combination LSTM \n Training time: {o} hours '.format(o=hours))
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.legend(['Comedy', 'Thriller'])
plt.show()
