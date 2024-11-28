import torch
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RnnModel(nn.Module):
    def __init__(self, input_feature, hidden_size_1, hidden_size_2, hidden_size_3, output_num_cnt = 1):
        super(RnnModel, self).__init__()
        self.input_feature = input_feature
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_num_cnt = output_num_cnt
        self.Linear1 = nn.Linear(self.input_feature, self.hidden_size_1, device=DEVICE)
        self.Relu1 = nn.Sigmoid()
        # self.RNN1 = nn.RNN(hidden_size_1, hidden_size_2, device=DEVICE)
        self.RNN1 = nn.LSTMCell(hidden_size_1, hidden_size_2, device=DEVICE)
        # self.tanh1 = nn.Tanh()
        self.Linear2 = nn.Linear(self.hidden_size_2, self.hidden_size_3, device=DEVICE)
        self.Relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(self.hidden_size_3, self.output_num_cnt, device=DEVICE)
        self.prev_hidden = torch.zeros(1,self.hidden_size_2, device=DEVICE)
        self.prev_c = torch.zeros(1, self.hidden_size_2, device=DEVICE)

    def forward(self, x):
        #do not consider batch
        T = x.shape[0]
        for i in range(T):
            assert (len(x.shape) == 2)
            output_linear = self.Linear1.forward(x[i])
            output_linear = self.Relu1.forward(output_linear)
            output_linear = output_linear.unsqueeze(0)
            # print(output_linear.shape)
            new_hidden, new_prev_c = self.RNN1.forward(output_linear, (self.prev_hidden, self.prev_c))
            # new_hidden = self.tanh1.forward(new_hidden)
            # print(new_hidden.shape)
            if i != T - 1:
                self.prev_hidden = new_hidden.detach()
                self.prev_c = new_prev_c.detach()
            else:
                self.prev_hidden = new_hidden
                self.prev_c = new_prev_c
                output = self.Linear2.forward(self.prev_hidden)
                output = self.Relu2.forward(output)
                output = self.Linear3.forward(output)
                output = output.squeeze()
        return output

    def init_prev_hidden(self):
        self.prev_hidden = torch.zeros(1, self.hidden_size_2, dtype=torch.float, device=DEVICE)

if __name__ == "__main__":
    loss_fn = nn.MSELoss()
    input = torch.randn(5, 10, 27)
    label = torch.randn(1)
    model = RnnModel(27, 12, 8, 5)

    opt = torch.optim.Adam(model.parameters(), 0.01)
    for epo in range(3):
        loss_total = 0
        for i in range(len(input)):
            output = model.forward(input[i])
            print(output.shape)
            model.zero_grad()
            loss = loss_fn(output, label)
            loss_total += loss.item()
            loss.backward()
            opt.step()
        print(loss_total)
