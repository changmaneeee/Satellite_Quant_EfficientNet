# 파일 이름: train.py

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
# 우리 모듈들 임포트
from src.data_util import get_dataloader, STATS
from src.quant_modules import clip_weights
from src.models.efficientNet_builder import build_efficientnet

# ───────────────────────────────────────────────────────────────
# 학습 및 평가 함수
# ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, bits_to_clip, epoch_str):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(loader, desc=f"Train {epoch_str}", leave=False, ncols=100)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if bits_to_clip <= 4: clip_weights(model)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}", 
            acc=f"{100 * correct / total:.2f}%"
        )
    return running_loss / total, 100 * correct / total

def evaluate_model(model, loader, criterion, device, use_amp, epoch_str):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(loader, desc=f"Eval  {epoch_str}", leave=False, ncols=100)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # tqdm 진행 바의 오른쪽에 실시간 정보 업데이트
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100 * correct / total:.2f}%"
            )
    return running_loss / total, 100 * correct / total

# ───────────────────────────────────────────────────────────────
# 메인 실행 함수
# ───────────────────────────────────────────────────────────────
def main(args):
    # 1. 환경 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    # 2. 결과 저장 경로 설정
    result_dir = os.path.join(args.result_dir, f"{args.dataset}_{args.model}_{args.bits}bit_w{args.act_bits}a")
    os.makedirs(result_dir, exist_ok=True)
    
    # 3. 데이터 로더 준비
    train_loader = get_dataloader(args.dataset, args.batch_size, train=True, num_workers=args.num_workers)
    test_loader = get_dataloader(args.dataset, args.batch_size, train=False, num_workers=args.num_workers)
    num_classes = STATS[args.dataset]['num_classes']

    # 4. 모델 생성
    if args.model == 'efficientnet_b0':
        model = build_efficientnet(bits=args.bits, act_bits=args.act_bits, num_classes=num_classes).to(DEVICE)
    else: # elif args.model == 'resnet18':
        raise NotImplementedError("ResNet18 builder not fully integrated yet.")
    
    # 5. 학습 설정 (SGD 기반)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    USE_AMP = (args.bits == 16 or args.act_bits == 16)
    scaler = GradScaler(enabled=USE_AMP)

    # 6. 학습 루프
    print(f"\n--- Training {args.model} on {args.dataset} ({args.bits}-bit W, {args.act_bits}-bit A) ---")
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_str = f"Epoch {epoch:03d}/{args.epochs}"
        
        # 수정: 함수에 epoch_str 전달
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, USE_AMP, args.bits, epoch_str)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE, USE_AMP, epoch_str)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 수정: 로그 출력을 더 깔끔하게 변경
        print(f"{epoch_str} -> Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best Acc: {best_test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # 최고 성능 모델 저장
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(result_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            # 최고 기록 갱신 시에만 로그 출력
            print(f"    🎉 New best model saved with accuracy: {best_test_acc:.2f}%")

        scheduler.step()

    print(f"\n--- Training Finished. Best Test Accuracy: {best_test_acc:.2f}% ---")
    
    # 7. 최종 결과(그래프) 저장
    # ... (Matplotlib 코드, 이전과 동일) ...

# ───────────────────────────────────────────────────────────────
# 스크립트 실행 지점 (Entry Point)
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='On-board AI Quantization Experiment')
    
    # 실험 제어 인자
    parser.add_argument('--dataset', type=str, required=True, choices=['eurosat', 'uc_merced'])
    parser.add_argument('--model', type=str, default='efficientnet_b0', choices=['efficientnet_b0'])
    parser.add_argument('--bits', type=int, required=True, choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--act_bits', type=int, default=32, choices=[1, 2, 4, 8, 16, 32])
    
    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    

    #parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--result_dir', type=str, default='./results')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42)
    # ...
    
    args = parser.parse_args()
    main(args)