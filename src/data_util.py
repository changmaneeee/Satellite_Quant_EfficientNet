# 파일 이름: src/data_utils.py (최종 버전)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# 데이터셋별 고유 정보 (평균, 표준편차, 클래스 수, 기본 이미지 크기)
STATS = {
    'eurosat': {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'num_classes': 10, 'size': 64
    },
    'uc_merced': { # UC Merced는 torchvision에 없으므로, 이전의 ImageFolder 방식 유지
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'num_classes': 21, 'size': 256
    }
    # 여기에 다른 데이터셋 추가 가능
}

def get_dataloader(dataset_name, batch_size, train=True, num_workers=4, train_split_ratio=0.8, seed=42):
    """
    지정된 데이터셋에 맞는 데이터로더를 생성하여 반환합니다.
    EuroSAT은 torchvision에서 직접 다운로드하고 분할합니다.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in STATS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Supported: {list(STATS.keys())}")

    info = STATS[dataset_name]
    mean, std = info['mean'], info['std']
    image_size = 224  # 모든 모델의 입력 크기를 224x224로 통일

    # 1. 데이터 변환(Transform) 정의
    if train:
        # 훈련 데이터용: 강력한 데이터 증강 적용
        transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        # 테스트 데이터용: 데이터 증강 없음
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # 2. 데이터셋 객체 생성
    if dataset_name == 'eurosat':
        # EuroSAT은 자체적인 train/test 분할이 없으므로, 전체 데이터셋을 다운로드하고 수동으로 분할
        full_dataset = datasets.EuroSAT(
            root='./data/EuroSAT', 
            download=True,  # True로 설정하면 데이터가 없을 때 자동으로 다운로드
            transform=transform
        )
        
        # 데이터셋을 train/test로 분할
        generator = torch.Generator().manual_seed(seed) # 재현성을 위한 시드 고정
        train_size = int(len(full_dataset) * train_split_ratio)
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
        
        # 현재 함수가 train용 로더를 원하는지, test용 로더를 원하는지에 따라 데이터셋 선택
        dataset = train_dataset if train else test_dataset
        
    elif dataset_name == 'uc_merced':
        # UC Merced는 이전처럼 ImageFolder를 사용하는 로직 유지 (폴더 준비 필요)
        #from .uc_merced_utils import prepare_uc_merced_folders, UCMercedDataset
        data_path = './data/UCMerced_LandUse'
        #prepare_uc_merced_folders(data_path, train_split_ratio=train_split_ratio, seed=seed)
        dataset_path = os.path.join(data_path, 'train' if train else 'test')
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        
    else:
        raise NotImplementedError

    # 3. 데이터로더 생성
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader