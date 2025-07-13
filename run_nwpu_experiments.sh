#!/bin/bash

# =======================================================
# NWPU-RESISC45 Quantization Experiment Runner
# =======================================================

# --- 실행 전략 선택 (sgd 또는 adamw) ---
if [ -z "$1" ]; then
    echo "Usage: ./run_nwpu_experiments.sh [sgd|adamw]"
    exit 1
fi
OPTIMIZER_STRATEGY=$1

# --- 실험 파라미터 정의 ---
MODEL_LIST=("resnet18") #"efficientnet_b0"
BITS_LIST=(1 2 4 8 16 32)
CLASSES_LIST=(10 20 30 45) # NWPU에 맞는 클래스 수
ACT_BITS_LIST=(32) 
EPOCHS=150
BATCH_SIZE=32

# 옵티마이저 전략에 따른 하이퍼파라미터 설정
if [ "$OPTIMIZER_STRATEGY" = "sgd" ]; then
    OPTIMIZER_ARGS="--optimizer sgd --momentum 0.9"
    WEIGHT_DECAY=5e-4
    LR_LOW_BIT=0.01
    LR_HIGH_BIT=0.1
elif [ "$OPTIMIZER_STRATEGY" = "adamw" ]; then
    OPTIMIZER_ARGS="--optimizer adamw"
    WEIGHT_DECAY=1e-4
    LR_LOW_BIT=0.0001
    LR_HIGH_BIT=0.0001
else
    echo "ERROR: Invalid optimizer strategy '$1'."
    exit 1
fi

# --- 전체 학습 실행 ---
for MODEL in "${MODEL_LIST[@]}"; do
    for CLASSES in "${CLASSES_LIST[@]}"; do
        for BITS in "${BITS_LIST[@]}"; do
            # ... (이전 run 스크립트의 for 루프 내부와 동일) ...
            
            if [ "$BITS" -le 4 ]; then
                LR=$LR_LOW_BIT
            else
                LR=$LR_HIGH_BIT
            fi

            # 로그 파일 이름에 'nwpu' 명시
            LOG_FILE="logs_nwpu/training/nwpu_${MODEL}_${BITS}b_w${ACT_BITS}a_${CLASSES}c.log"
            
            echo "--- TRAINING: NWPU, ${MODEL}, ${CLASSES}-class, ${BITS}-bit ---"
            
            # train_nwpu.py 호출
            python train_nwpu.py \
                --model "$MODEL" \
                --num_classes "$CLASSES" \
                --bits "$BITS" \
                --act_bits ${ACT_BITS_LIST[0]} \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                $OPTIMIZER_ARGS \
                --lr "$LR" \
                --weight_decay "$WEIGHT_DECAY" \
                2>&1 | tee "$LOG_FILE"
            
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