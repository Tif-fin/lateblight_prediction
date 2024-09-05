import torch 
from torch_geometric_temporal.nn.attention.stgcn import STConv



class STGCN(torch.nn.Module):
    """
    Processes a sequence of graph data to produce a spatio-temporal embedding
    to be used for regression, classification, clustering, etc.
    """
    def __init__(self):
        super(STGCN, self).__init__()
        # self.stconv_block1 = STConv(210, 9, 18, 32, 2, 4)
        # self.stconv_block2 = STConv(210, 32, 18, 9, 2, 4)
        # self.fc = torch.nn.Linear(9, 4)
        self.stconv_block1 = STConv(1359, 6, 64, 128, 34, 15)
        self.stconv_block2 = STConv(1359, 128, 256, 64, 7, 15)
        self.stconv_block3 = STConv(1359, 64, 32, 16, 7, 3)
        self.fc = torch.nn.Linear(16, 3)
    # def forward(self, x, edge_index, edge_attr):
    #     temp = self.stconv_block1(x, edge_index)
    #     temp = self.stconv_block2(temp, edge_index)
    #     temp = self.fc(temp)
        
    #     return temp
    
    def forward(self, x, edge_index, edge_attr):
        temp = self.stconv_block1(x, edge_index, edge_attr)
        temp = self.stconv_block2(temp, edge_index, edge_attr)
        temp = self.stconv_block3(temp, edge_index, edge_attr)
        temp = self.fc(temp)

        return temp

model = STGCN()
# model.load_weight('weights.pth')
# weight = torch.load('static/Model_60Lags_STConv_Best_Feb18.pt')
weight = torch.load('static/30_days.pth')
model.load_state_dict(weight)

model.predict()



