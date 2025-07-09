# 파일 이름: src/data_utils.py (최종 수정본)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# 데이터셋별 고유 정보
STATS = {
    'eurosat': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'num_classes': 10},
    'uc_merced': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'num_classes': 21}
}

def prepare_folders_from_raw(dataset_name, data_path='./data', train_split=0.8, seed=42):
    """
    ./data/{dataset_name}/raw/ 폴더의 이미지들을 ./data/{dataset_name}/train/ 및 ./data/{dataset_name}/test/로 분할.
    최초 1회만 실행됨.
    """
    raw_path = os.path.join(data_path, dataset_name, 'raw')
    train_path = os.path.join(data_path, dataset_name, 'train')
    test_path = os.path.join(data_path, dataset_name, 'test')

    # raw 폴더가 없으면, 이미 분할되었다고 가정하고 함수 종료
    if not os.path.exists(raw_path):
        if not os.path.exists(train_path):
             raise FileNotFoundError(f"Neither 'raw' nor 'train' directory found for '{dataset_name}' in {data_path}.")
        return

    # train/test 폴더가 이미 있으면, 작업이 완료된 것으로 보고 함수 종료
    if os.path.exists(train_path) and os.path.exists(test_path):
        return

    print(f"INFO: Preparing train/test split for '{dataset_name}' from '{raw_path}'...")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    class_names = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
    for class_name in class_names:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
        
        all_files = glob.glob(os.path.join(raw_path, class_name, '*.*')) # .tif, .jpg 등 모든 확장자
        if not all_files: continue

        train_files, test_files = train_test_split(all_files, train_size=train_split, random_state=seed, shuffle=True)
        
        for f in train_files: shutil.copy(f, os.path.join(train_path, class_name))
        for f in test_files: shutil.copy(f, os.path.join(test_path, class_name))
        
    print(f"INFO: Preparation complete for '{dataset_name}'.")

def get_dataloader(dataset_name, batch_size, train=True, num_workers=4, image_size=224, seed=42):
    dataset_name = dataset_name.lower()
    if dataset_name not in STATS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    info = STATS[dataset_name]
    mean, std = info['mean'], info['std']

    if train:
        transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    data_path = os.path.join('./data', dataset_name)
    # 데이터 로딩 전, 폴더가 준비되었는지 확인 및 분할 (핵심!)
    prepare_folders_from_raw(dataset_name, './data', seed=seed)

    dataset_path = os.path.join(data_path, 'train' if train else 'test')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}. Please check your data folder structure.")

    # ImageFolder는 정리된 폴더 구조를 기반으로 데이터셋을 생성
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return loader