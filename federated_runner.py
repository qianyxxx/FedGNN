# 合并后的 federated_runner.py
import torch
from copy import deepcopy
from models.gnn_model import GCN
from client import Client
from server import Server
from data.data_loader import DataLoader as CustomDataLoader
from torch_geometric.data import DataLoader as GeoDataLoader
import torch.nn.functional as F

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            data.to(device)
            output = model(data.x.to(device), data.edge_index.to(device))
            pred = output.argmax(dim=1)
            correct += (pred == data.y.to(device)).sum().item()  # 确保标签也在正确的设备上
            total += data.y.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')


def run_federated_learning(num_rounds, num_clients, batch_size, dataset_path, dataset_name, epochs):
    num_node_features = 1433  # 根据实际数据集调整
    num_classes = 7
    
    global_model = GCN(num_node_features, num_classes)
    loader = CustomDataLoader(dataset_path)
    test_loader = loader.get_dataloader(dataset_name, batch_size, shuffle=False)  # 创建测试数据加载器

    clients = []
    for i in range(num_clients):
        dataloader = loader.get_dataloader(dataset_name, batch_size, shuffle=True)
        client_model = deepcopy(global_model)
        client = Client(i, client_model, dataloader, torch.optim.Adam, F.nll_loss)
        clients.append(client)
    
    server = Server(deepcopy(global_model))

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        client_models = []
        for client in clients:
            client_model_state_dict = client.train(epochs=epochs)
            client_models.append(client_model_state_dict)
        
        global_model = server.update_global_model(client_models)
        evaluate_model(global_model, test_loader, device='cpu')  # 在每轮结束后评估模型

    torch.save(global_model.state_dict(), f'{dataset_path}/global_model.pth')
