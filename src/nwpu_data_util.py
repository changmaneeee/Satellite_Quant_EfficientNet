# 파일 이름: src/nwpu_data_utils.py

import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# NWPU 데이터셋 정보
STATS = {'mean': (0.368, 0.381, 0.344), 'std': (0.145, 0.136, 0.131)} # NWPU 데이터셋 논문에서 계산된 값
NUM_TOTAL_CLASSES = 45

# 클래스 분할 정보를 저장할 파일 경로
CONFIG_PATH = "./configs/nwpu_class_splits.json"

def prepare_class_splits(seed=42):
    """
    NWPU 45개 클래스를 무작위로 섞어 서브셋 정보를 JSON 파일에 저장합니다.
    파일이 없으면 새로 생성합니다.
    """
    if os.path.exists(CONFIG_PATH):
        # print("INFO: NWPU class split file already exists.")
        return

    print(f"INFO: Creating new class splits for NWPU dataset...")
    np.random.seed(seed)
    all_class_indices = list(range(NUM_TOTAL_CLASSES))
    np.random.shuffle(all_class_indices)

    # 10, 20, 30, 45개 클래스 서브셋의 '인덱스'를 저장
    class_splits = {
        '10': all_class_indices[:10],
        '20': all_class_indices[:20],
        '30': all_class_indices[:30],
        '45': all_class_indices
    }
    
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(class_splits, f, indent=4)
    print(f"NWPU class splits saved to {CONFIG_PATH}")


def get_nwpu_dataloader(num_classes, batch_size, train=True, num_workers=4, image_size=224):
    """NWPU-RESISC45 데이터셋의 서브셋에 대한 데이터로더를 반환합니다."""

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.TrivialAugmentWide() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(STATS['mean'], STATS['std']),
        transforms.RandomErasing(p=0.25) if train else transforms.Lambda(lambda x: x),
    ])

    data_path = os.path.join('./data', 'NWPU-RESISC45', 'train' if train else 'test')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}. Please split NWPU data into train/test folders first.")
    
    # ImageFolder는 폴더 이름을 알파벳 순으로 정렬하여 0, 1, 2... 라벨을 자동 할당
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    if num_classes == NUM_TOTAL_CLASSES:
        
        loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
        return loader, None
    # 서브셋 생성 로직
    prepare_class_splits() # 파일이 없으면 생성
    with open(CONFIG_PATH, 'r') as f:
        class_splits = json.load(f)

    selected_class_indices = class_splits.get(str(num_classes))
    if selected_class_indices is None:
        raise KeyError(f"Number of classes '{num_classes}' not defined in {CONFIG_PATH}.")

    # 전체 데이터셋에서, 선택된 클래스 인덱스에 해당하는 데이터만 필터링
    # `full_dataset.targets`는 ImageFolder가 할당한 0~44 사이의 정수 라벨 리스트
    indices_to_keep = [i for i, target in enumerate(full_dataset.targets) if target in selected_class_indices]
    
    subset = Subset(full_dataset, indices_to_keep)
    
    # 중요: 현재 subset의 라벨은 0~44 사이의 값들입니다.
    # 모델의 마지막 레이어 출력 뉴런 수가 num_classes와 일치해야 하므로,
    # 손실 함수(CrossEntropyLoss)가 이 라벨들을 처리할 수 있는지 확인이 필요합니다.
    # CrossEntropyLoss는 라벨이 [0, num_classes-1] 범위를 벗어나면 에러를 발생시킵니다.
    # 따라서, 라벨 리매핑이 필요하지만, 여기서는 train.py에서 처리하도록 유연성을 둡니다.
    
    loader = DataLoader(subset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return loader, {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_class_indices)}