# federated/server.py
import torch

class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    def update_global_model(self, client_models):
        global_state_dict = self.global_model.state_dict()
        
        # 计算平均模型参数
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.mean(torch.stack([client_models[i][key] for i in range(len(client_models))]), 0)
        
        self.global_model.load_state_dict(global_state_dict)
        return self.global_model
