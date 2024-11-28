import os.path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.onnx.symbolic_opset9 import tensor
from sklearn import preprocessing
from model import *

need_train = False
test_cnt = 3
train_set = []
test_set = []
T = 10
EPOCH_CNT = 8000
best_model_score = 15
best_model_score_ori = best_model_score
loss_fn = nn.MSELoss()
loss_fn.to(DEVICE)

def train_predict(need_train=True, repeat_cnt = 10, hidden_size_1 = 4, hidden_size_2 = 3, hidden_size_3 = 2):
    # frame = pd.read_csv('sin_test.csv')
    frame = pd.read_csv('stock.csv')
    cnt_total = frame.shape[0]
    test_set = []
    global best_model_score
    if test_cnt == 0:
        test_valid = []
        test_base = frame.iloc[0:T,:]
        test_base = test_base.iloc[::-1].reset_index(drop=True)

        # 对于sin_test的数据，这个会有严重问题
        test_base = preprocessing.StandardScaler().fit_transform(test_base)
        test_set.append((torch.tensor(np.array(test_base), dtype=torch.float, device=DEVICE),
                         torch.tensor(np.array(test_valid), dtype=torch.float,device=DEVICE)))
    else:
        for i in range(test_cnt, 0, -1):
            test_valid = frame.iloc[i - 1].loc['change']
            test_base = frame.iloc[i + T - 1 : i - 1 : -1,:]
            test_base = preprocessing.StandardScaler().fit_transform(test_base)
            test_set.append((torch.tensor(np.array(test_base), dtype=torch.float, device=DEVICE),
                             torch.tensor(np.array(test_valid), dtype=torch.float,device=DEVICE)))



    input_feature = test_base.shape[1]

    train_set = []
    for i in range(cnt_total - T - 1, test_cnt - 1, -1):
        train_valid = frame.iloc[i].loc['change']
        train_base = frame.iloc[i + T : i : -1,:]
        train_base = preprocessing.StandardScaler().fit_transform(train_base)
        train_set.append((torch.tensor(np.array(train_base), dtype=torch.float, device=DEVICE),
                          torch.tensor(np.array(train_valid), dtype=torch.float, device=DEVICE)))

    if need_train:
        ignore_pre_train = False
        min_loss_all_repeat = 10000000
        for rept in range(repeat_cnt):
            print(f"===============第{rept}次重复================")

            if not os.path.exists('./rnn_stock.bak') and not ignore_pre_train:
                LR = 0.06
                use_pre_train = False
                model = RnnModel(input_feature, hidden_size_1, hidden_size_2, hidden_size_3,1)
            else:
                LR = 0.04
                use_pre_train = True
                model = torch.load("rnn_stock.bak")
            model.to(DEVICE)
            opt = torch.optim.SGD(model.parameters(), LR)

            model.train()
            last_epo_loss = 1000
            for epo in range(EPOCH_CNT):
                total_loss = 0
                for one_base, one_valid in train_set:
                    model.init_prev_hidden()
                    result = model.forward(one_base)
                    loss = loss_fn(result, one_valid)
                    total_loss += loss.item()

                    loss.backward()
                    opt.step()
                    model.zero_grad()
                if epo % 50 == 0:
                    print(f"第{epo}轮训练，loss为： ", total_loss)
                    if total_loss <= best_model_score / 2.0:
                        torch.save(model, "rnn_stock.bak")
                        break
                    if epo >= 100 and last_epo_loss == total_loss:
                        print(total_loss, last_epo_loss, last_epo_loss - total_loss, total_loss * 0.001)
                        break
                    if epo == 500 and total_loss > 30:
                        print("降低太慢，第500轮loss大于30.")
                        break
                    if epo == 1000 and total_loss > 15:
                        print("降低太慢，第1000轮loss大于10.")
                        break
                    if epo == 2000 and total_loss > 3:
                        print("降低太慢，第2000轮loss大于3.")
                        break
                    if last_epo_loss - total_loss < total_loss*0.005:
                        if total_loss > 10 and LR < 0.02:
                            break
                        # print(total_loss, last_epo_loss, last_epo_loss - total_loss, total_loss*0.001)
                        if LR > 0.01:
                            LR = LR / 1.2
                        else:
                            LR = LR / 1.8
                        print(f"++++++++++++++  using LR {LR} +++++++++++++++")
                        if LR < 0.001:
                            break
                        for param_group in opt.param_groups:
                            param_group['lr'] = LR
                    last_epo_loss = total_loss
                # if epo % 20 == 0:
                #     for k, v in model.named_parameters():
                #         print(k, v)
            if total_loss < min_loss_all_repeat:
                min_loss_all_repeat = total_loss
                torch.save(model, "rnn_stock")
                if total_loss < best_model_score:
                    torch.save(model, "rnn_stock.bak")
                    best_model_score = total_loss
            if total_loss <= best_model_score_ori:
                break
            elif use_pre_train:
                ignore_pre_train = True
                best_model_score = best_model_score_ori
    model = torch.load("rnn_stock")
    model.eval()
    model.init_prev_hidden()

    last = train_set[-1][1]
    start = torch.Tensor.cpu(last).detach().numpy()
    one_test_base, one_test_valid = test_set[0]
    result = model.forward(one_test_base)
    # plt.plot(draw_pred, label='pred')
    # plt.plot(draw_valid, label='valid')
    # plt.legend()
    # plt.show()
    return torch.Tensor.cpu(result).detach().numpy(), torch.Tensor.cpu(one_test_valid).detach().numpy()

if __name__ == "__main__":
    train_predict(need_train)