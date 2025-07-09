# 파일 이름: plot_results.py

import os
import re # 정규 표현식을 사용한 문자열 파싱을 위해
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 더 예쁜 그래프를 위해
import re

# ───────────────────────────────────────────────────────────────
# 1. 로그 파일 파싱 및 데이터프레임 생성
# ───────────────────────────────────────────────────────────────
def parse_log_file(filepath):
    """
    하나의 로그 파일을 파싱하여 에포크별 데이터를 추출합니다.
    새로운 로그 형식에 맞게 수정되었습니다.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: Log file not found at {filepath}, skipping.")
        return None

    # 새로운 로그 형식에 맞는 정규 표현식
    # 예: Epoch 070/200 -> Train Acc: 85.32% | Test Acc: 75.17% | ...
    # 우리는 Train Acc와 Test Acc만 추출하면 됩니다. Loss는 이 라인에 없으므로 무시.
    pattern = re.compile(
        r"Epoch (\d+)/\d+ -> "
        r"Train Acc: ([\d.]+)% \| "
        r"Test Acc: ([\d.]+)%"
    )
    
    records = []
    # findall은 파일 전체에서 모든 매칭되는 부분을 찾아 리스트로 반환
    matches = pattern.findall(content)

    if not matches:
        # 패턴에 맞는 라인을 하나도 찾지 못했을 경우
        # 아마도 학습이 시작되자마자 에러가 났을 수 있음
        print(f"Warning: No valid epoch data found in {filepath}. Skipping.")
        return None

    for match in matches:
        # match는 ('70', '85.32', '75.17') 와 같은 튜플
        epoch, train_acc, test_acc = match
        records.append({
            'epoch': int(epoch),
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            # 로그에 Loss 정보가 없으므로, 필요하다면 0 또는 NaN으로 채움
            'train_loss': 0.0, 
            'test_loss': 0.0
        })
            
    return pd.DataFrame(records)

# plot_results.py (디버깅 코드가 추가된 버전)

def load_all_logs(log_dir='./logs/training'):
    """지정된 폴더의 모든 로그 파일을 읽어 하나의 데이터프레임으로 합칩니다."""
    all_dfs = []
    
    print(f"--- DEBUG: Scanning directory -> {os.path.abspath(log_dir)}")
    if not os.path.exists(log_dir):
        print(f"ERROR: Log directory not found at {log_dir}")
        return None

    # os.listdir()가 반환하는 모든 항목을 먼저 출력해본다.
    try:
        all_items_in_dir = os.listdir(log_dir)
        print(f"--- DEBUG: Items found in directory: {all_items_in_dir}")
    except Exception as e:
        print(f"ERROR: Could not list directory {log_dir}: {e}")
        return None

    for filename in os.listdir(log_dir):
        if not filename.endswith(".log"):
            continue

        filepath = os.path.join(log_dir, filename)
        
        try:
            # ==================== 수정 시작 ====================
            clean_filename = filename.replace('.log', '')
            parts = clean_filename.split('_')
            
            # 모델 이름이 'efficientnet_b0' 처럼 2 파트로 나뉘는 것을 처리
            # 파일 이름 구조: [dataset, model_part1, model_part2, ..., bits, act_bits]
            
            # 뒤에서부터 파싱하는 것이 더 안정적임
            act_bits_str = parts[-1]
            bits_str = parts[-2]
            
            # 데이터셋은 첫 번째 파트
            dataset = parts[0]
            
            # 모델 이름은 dataset과 bits/act_bits 사이의 모든 파트를 합침
            model = "_".join(parts[1:-2])

            bits_match = re.search(r'(\d+)b', bits_str)
            act_bits_match = re.search(r'(\d+)a', act_bits_str)

            if not bits_match or not act_bits_match:
                print(f"Warning: Could not parse bits/act_bits from filename: {filename}. Skipping.")
                continue
                
            bits = int(bits_match.group(1))
            act_bits = int(act_bits_match.group(1))
            # ===================== 수정 끝 =====================

        except (IndexError, ValueError) as e:
            print(f"Warning: Error parsing filename '{filename}': {e}. Skipping.")
            continue

        df = parse_log_file(filepath)
        
        if df is not None and not df.empty:
            df['dataset'] = dataset
            df['model'] = model
            df['bits'] = bits
            df['act_bits'] = act_bits
            all_dfs.append(df)
            
    if not all_dfs:
        print("No valid log files found or parsed. Exiting.")
        return None

    return pd.concat(all_dfs, ignore_index=True)

# ───────────────────────────────────────────────────────────────
# 2. 그래프 생성 함수
# ───────────────────────────────────────────────────────────────
def plot_individual_training_curves(df, output_dir='./results/training_plots'):
    """각 개별 실험에 대한 학습 곡선 그래프를 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 실험 조건별로 그룹화
    for (dataset, model, bits, act_bits), group_df in df.groupby(['dataset', 'model', 'bits', 'act_bits']):
        
        plt.figure(figsize=(14, 6))
        title = f'{model} on {dataset} ({bits}-bit W, {act_bits}-bit A)'
        plt.suptitle(title, fontsize=16)

        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(group_df['epoch'], group_df['train_loss'], label='Train Loss')
        plt.plot(group_df['epoch'], group_df['test_loss'], label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.grid(True)
        plt.legend()

        # Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(group_df['epoch'], group_df['train_acc'], label='Train Accuracy')
        plt.plot(group_df['epoch'], group_df['test_acc'], label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100) # Y축을 0~100으로 고정
        plt.title('Accuracy Curve')
        plt.grid(True)
        plt.legend()
        
        # 파일 저장
        filename = f"{dataset}_{model}_{bits}b_w{act_bits}a.png"
        save_path = os.path.join(output_dir, filename)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close() # 메모리 해제를 위해 닫아줌
        
    print(f"Individual training curves saved to {output_dir}")

def plot_final_accuracy_comparison(df, output_dir='./results'):
    """
    모든 실험의 최종(최고) 테스트 정확도를 비교하는 막대 그래프를 생성합니다.
    """
    # 각 실험 조건별로 마지막 에포크의 test_acc (또는 최고 test_acc)를 찾음
    # 여기서는 간편하게 마지막 에포크의 값으로 계산
    final_results = df.loc[df.groupby(['dataset', 'model', 'bits', 'act_bits'])['epoch'].idxmax()]
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Seaborn을 사용하여 더 예쁜 막대 그래프 생성
    ax = sns.barplot(data=final_results, x='bits', y='test_acc', hue='dataset', palette='viridis')
    
    plt.title('Final Test Accuracy vs. Bit-width', fontsize=18)
    plt.xlabel('Weight Bit-width', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.ylim(min(50, final_results['test_acc'].min() - 5), 100)
    plt.legend(title='Dataset')

    # 막대 위에 정확도 수치 표시
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    save_path = os.path.join(output_dir, 'final_accuracy_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Final accuracy comparison plot saved to {save_path}")

# ───────────────────────────────────────────────────────────────
# 3. 메인 실행 지점
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. 모든 로그 파일 로드 및 파싱
    full_df = load_all_logs()
    
    if full_df is not None:
        # 2. 개별 학습 곡선 그래프 생성
        plot_individual_training_curves(full_df)
        
        # 3. 최종 성능 비교 그래프 생성
        plot_final_accuracy_comparison(full_df)

        print("\nAll plots have been generated successfully!")