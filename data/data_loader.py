# data_loader.py
import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import DataLoader as GeoDataLoader

# 定义数据集的存储路径
dataset_root = ''
raw_dir = os.path.join(dataset_root, 'raw')
processed_dir = os.path.join(dataset_root, 'processed')

# 确保目录存在
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

class DataLoader:
    def __init__(self, root_dir, datasets=['Cora', 'CiteSeer', 'PubMed']):
        self.root_dir = root_dir
        self.datasets = datasets

    def download_and_process(self):
        for dataset_name in self.datasets:
            # 指定数据集的下载和存储路径
            dataset_path = os.path.join(self.root_dir, dataset_name)
            # 使用NormalizeFeatures转换
            dataset = Planetoid(root=dataset_path, name=dataset_name, transform=NormalizeFeatures())
            
            # 数据下载和预处理
            print(f'Downloading and processing {dataset_name}...')
            _ = dataset[0]  # 触发下载和预处理
            
            # 在这里，数据集自动下载到指定的root目录下的raw子目录，并在处理后保存在processed子目录
            print(f'{dataset_name} has been downloaded and processed.')
            print(f'Raw data path: {os.path.join(dataset_path, "raw")}')
            print(f'Processed data path: {os.path.join(dataset_path, "processed")}')
            
            # 这里可以添加额外的数据处理逻辑，并将处理后的数据保存到processed_dir

    def test_data(self):
        for dataset_name in self.datasets:
            dataset_path = os.path.join(self.root_dir, dataset_name)
            dataset = Planetoid(root=dataset_path, name=dataset_name, transform=NormalizeFeatures())
            
            # 加载数据
            data = dataset[0]  # 加载第一个图
            
            # 打印数据集的基本信息
            print(f'\n{dataset_name} Dataset:')
            print(f'Number of nodes: {data.num_nodes}')
            print(f'Number of edges: {data.num_edges}')
            print(f'Number of features per node: {data.num_node_features}')
            print(f'Number of classes: {dataset.num_classes}')
            print(f'Number of training nodes: {data.train_mask.sum()}')
            print(f'Number of validation nodes: {data.val_mask.sum()}')
            print(f'Number of test nodes: {data.test_mask.sum()}')
            
            classes = data.y.unique().tolist()
            print(f'Classes: {classes}')
            # 为了简化输出，这里我们不展示边的具体信息，但是你可以通过data.edge_index来访问它
    


    def get_dataloader(self, dataset_name, batch_size=32, shuffle=True):
        dataset_path = os.path.join(self.root_dir, dataset_name)
        dataset = Planetoid(root=dataset_path, name=dataset_name, transform=NormalizeFeatures())

        # 正确创建 PyTorch Geometric DataLoader
        return GeoDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    loader = DataLoader(dataset_root)
    loader.download_and_process()
    loader.test_data()
