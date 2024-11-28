import numpy as np
import train_and_pred_stock as stock
import train_and_pred_lot as lot
import matplotlib.pyplot as plt

pred_cnt = 20

result_pred = []
result_valid = []
for i in range(pred_cnt, -1 , -1):
    print(f" ####################  基线位置 {i}  #############",'\n')
    stock.test_cnt = i
    pred, valid = stock.train_predict()
    result_pred.append(pred)
    print(valid)
    if (isinstance(valid, np.ndarray) and valid.size == 0):
        continue
    else:
        result_valid.append(valid)
    print("pred: ",pred,"valid: ", valid)
plt.plot(result_pred, label='pred')
print(result_valid)
plt.plot(np.array(result_valid), label='valid')
plt.legend()
plt.show()
