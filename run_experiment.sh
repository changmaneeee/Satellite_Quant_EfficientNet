#!/bin/bash

# =================================================================
# Onboard-AI-Quantization: Master Experiment Script
# =================================================================

# --- 1. 실험 환경 설정 ---
echo "INFO: Setting up experiment environment..."
# 결과를 저장할 최상위 폴더들 생성
mkdir -p results models logs/training logs/benchmarking

# --- 2. 실험 파라미터 정의 ---
# 실험할 조합들을 배열로 정의합니다.
# 특정 실험만 하고 싶으면, 이 리스트를 수정하면 됩니다.
DATASET_LIST=("eurosat") # "uc_merced"도 추가 가능
MODEL_LIST=("efficientnet_b0") # "resnet18"도 추가 가능
BITS_LIST=(1 2 4 8 16 32)
# 활성화 양자화는 가중치와 동일하게 설정 (WnA n) 또는 32(Wn A32)로 실험
# 여기서는 가중치만 양자화하는 경우(act_bits=32)를 먼저 실행
ACT_BITS_LIST=(32) 

# 공통 하이퍼파라미터
EPOCHS=150
BATCH_SIZE=32
LR=0.0001  # SGD에 맞는 초기 학습률(0.05 / adamw에서는 1e-4)
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
NUM_WORKERS=4

# --- 3. 전체 학습 실행 ---
echo ""
echo "==================================================="
echo "  PHASE 1: STARTING BATCH TRAINING"
echo "==================================================="

# 4중 for 루프로 모든 조합에 대해 학습 실행
for DATASET in "${DATASET_LIST[@]}"; do
    for MODEL in "${MODEL_LIST[@]}"; do
        for BITS in "${BITS_LIST[@]}"; do
            for ACT_BITS in "${ACT_BITS_LIST[@]}"; do

                # WnA n 비트 실험에서 가중치와 활성화 비트가 다르면 건너뛰기 (선택사항)
                # if [ "$BITS" -ne 32 ] && [ "$ACT_BITS" -ne 32 ] && [ "$BITS" -ne "$ACT_BITS" ]; then
                #     continue
                # fi
                # ==================== 수정 1: 차등 학습률 설정 ====================
                # 비트 수에 따라 학습률(LR)을 동적으로 결정합니다.
                # if [ "$BITS" -le 4 ]; then
                #     # 1, 2, 4-bit: 학습이 불안정하므로 매우 낮은 학습률을 사용합니다.
                #     # AdamW에서 잘 동작했던 1e-4 (0.0001) 또는 더 낮은 값을 시도해볼 수 있습니다.
                #     # 여기서는 SGD에 맞춰서 조금 더 높은 0.005로 시작해봅니다.
                #     LR=0.005 
                # elif [ "$BITS" -eq 8 ]; then
                #     # 8-bit: 비교적 안정적이므로 중간 정도의 학습률을 사용합니다.
                #     LR=0.01
                # else # 16, 32-bit
                #     # 고정밀도 모델은 높은 초기 학습률로 시작해도 좋습니다.
                #     LR=0.05
                # fi
                # =================================================================                
                # 로그 파일 이름 생성
                LOG_FILE="logs/training/${DATASET}_${MODEL}_${BITS}b_w${ACT_BITS}a.log"
                
                echo ""
                echo "---------------------------------------------------"
                echo ">>> TRAINING: ${DATASET}, ${MODEL}, ${BITS}-bit W, ${ACT_BITS}-bit A"
                echo "    Log file: ${LOG_FILE}"
                echo "---------------------------------------------------"
#--momentum "$MOMENTUM" \                
                # train.py 스크립트 실행
                python train.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --bits "$BITS" \
                    --act_bits "$ACT_BITS" \
                    --epochs "$EPOCHS" \
                    --batch_size "$BATCH_SIZE" \
                    --lr "$LR" \
                    --weight_decay "$WEIGHT_DECAY" \
                    --num_workers "$NUM_WORKERS" \
                    2>&1 | tee "$LOG_FILE"
                
                # 에러 발생 시 스크립트 중단
                if [ $? -ne 0 ]; then
                    echo "!!!!!! ERROR: Training failed for ${DATASET}, ${MODEL}, ${BITS}-bit W, ${ACT_BITS}-bit A. !!!!!!"
                    echo "Please check the log file: ${LOG_FILE}"
                    exit 1
                fi
                
                echo ">>> FINISHED: ${DATASET}, ${MODEL}, ${BITS}-bit W, ${ACT_BITS}-bit A"
                
            done
        done
    done
done

echo ""
echo "==================================================="
echo "  ALL TRAINING EXPERIMENTS FINISHED SUCCESSFULLY!"
echo "==================================================="

# --- 4. 전체 벤치마킹 실행 (필요시 주석 해제) ---
# echo ""
# echo "==================================================="
# echo "  PHASE 2: STARTING BATCH BENCHMARKING"
# echo "==================================================="
# bash run_benchmarking.sh