# 파일 이름: train_nwpu.py

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
from src.nwpu_data_util import get_nwpu_dataloader # NWPU 전용 데이터 로더
from src.models.efficientNet_builder import build_efficientnet
from src.resnet_builder import build_resnet
from src.quant_modules import clip_weights

# ───────────────────────────────────────────────────────────────
# 학습 및 평가 함수 (라벨 리매핑 기능 추가)
# ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, bits_to_clip, epoch_str, label_map=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(loader, desc=f"Train {epoch_str}", leave=False, ncols=100)

    for inputs, labels in progress_bar:
        # ==================== 라벨 리매핑 추가 ====================
        if label_map:
            # 배치 내의 모든 라벨을 맵에 따라 변환
            labels = torch.tensor([label_map[l.item()] for l in labels], dtype=torch.long)
        # =======================================================
            
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

def evaluate_model(model, loader, criterion, device, use_amp, epoch_str, label_map=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(loader, desc=f"Eval  {epoch_str}", leave=False, ncols=100)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            # ==================== 라벨 리매핑 추가 ====================
            if label_map:
                labels = torch.tensor([label_map[l.item()] for l in labels], dtype=torch.long)
            # =======================================================
            
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total

# ───────────────────────────────────────────────────────────────
# 메인 실행 함수
# ───────────────────────────────────────────────────────────────
def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", DEVICE)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    result_dir = os.path.join(args.result_dir, f"nwpu_{args.model}_{args.bits}bit_w{args.act_bits}a_{args.num_classes}c")
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n--- Loading NWPU Dataset ({args.num_classes} classes) ---")
    train_loader, label_map = get_nwpu_dataloader(args.num_classes, args.batch_size, train=True, num_workers=args.num_workers)
    test_loader, _ = get_nwpu_dataloader(args.num_classes, args.batch_size, train=False, num_workers=args.num_workers)
    print(f"Train loader size: {len(train_loader.dataset)}, Test loader size: {len(test_loader.dataset)}")
    
    if args.model == 'efficientnet_b0':
        model = build_efficientnet(bits=args.bits, act_bits=args.act_bits, num_classes=args.num_classes).to(DEVICE)
    elif args.model == 'resnet18':
        model = build_resnet(bits=args.bits, act_bits=args.act_bits, num_classes=args.num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    USE_AMP = (args.bits == 16 or args.act_bits == 16)
    scaler = GradScaler(enabled=USE_AMP)

    print(f"\n--- Training {args.model} on NWPU ({args.bits}-bit W, {args.act_bits}-bit A, {args.num_classes} classes) ---")
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_str = f"Epoch {epoch:03d}/{args.epochs}"
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, USE_AMP, args.bits, epoch_str, label_map)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE, USE_AMP, epoch_str, label_map)
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss); history['test_acc'].append(test_acc)
        
        print(f"{epoch_str} -> Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best Acc: {best_test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(result_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"    🎉 New best model saved with accuracy: {best_test_acc:.2f}%")

        scheduler.step()

    total_time = time.time() - start_time
    print(f"\n--- Training Finished in {total_time/60:.2f} minutes. Best Test Accuracy: {best_test_acc:.2f}% ---")
    
    # 그래프 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['test_loss'], label='Test Loss'); plt.legend(); plt.grid(True); plt.title('Loss Curve')
    plt.subplot(1, 2, 2); plt.plot(history['train_acc'], label='Train Accuracy'); plt.plot(history['test_acc'], label='Test Accuracy'); plt.legend(); plt.grid(True); plt.title('Accuracy Curve')
    plot_path = os.path.join(result_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curves saved to {plot_path}")

# ───────────────────────────────────────────────────────────────
# 스크립트 실행 지점 (Entry Point)
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NWPU-RESISC45 Quantization Experiment')
    
    parser.add_argument('--model', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'resnet18'])
    parser.add_argument('--num_classes', type=int, required=True, choices=[10, 20, 30, 45])
    parser.add_argument('--bits', type=int, required=True, choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--act_bits', type=int, default=32, choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=min(os.cpu_count(), 8))
    parser.add_argument('--result_dir', type=str, default='./results')
    
    args = parser.parse_args()
    main(args)