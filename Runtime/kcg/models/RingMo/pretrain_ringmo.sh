# 环境变量设置
export HCCL_CONNECT_TIMEOUT=3000
export ASCEND_LAUNCH_BLOCKING=1

# 数据与路径配置
DATA_PATH_TRAIN=/code2/data_swin/AID.json
DATA_PATH_VAL=/code2/data_swin/AID.json
CHECKPOINT_PATH=/code2/ckpt
TENSOR_BOARD_DIR=/code2/logs

# 模型与训练参数
RINGMO_ARGS="
    --vision-pretraining-type dino \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --patch-dim 4 \
    --seq-length 3136 \
    --max-position-embeddings 3136 \
    --img-h 192 \
    --img-w 192 \
    --mask-factor 1.0 \
    --fp16 \
    --train-iters 5 \
    --lr-decay-style cosine \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.0005 \
    --min-lr 0.00001 \
    --attention-dropout 0.0 \
    --weight-decay 0.05 \
    --lr-warmup-iters 1 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce
"

# 分布式配置
NNODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 分布式主节点ip
# MASTER_ADDR=192.168.0.219
MASTER_ADDR=127.0.0.1
MASTER_PORT=22222

# 在不同节点上运行脚本要改下这个
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_ARGS="
    --dataloader-type external
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --data-path  $DATA_PATH_TRAIN $DATA_PATH_VAL\
    --no-data-sharding \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1 \
    --eval-iters 1 \
    --tensorboard-dir ${TENSOR_BOARD_DIR} \
"

if [ -d "$TENSOR_BOARD_DIR" ]; then
    echo "Cleaning TensorBoard directory: $TENSOR_BOARD_DIR"
    rm -rf "${TENSOR_BOARD_DIR}"/*
else
    echo "Creating TensorBoard directory: $TENSOR_BOARD_DIR"
    mkdir -p "$TENSOR_BOARD_DIR"
fi

torchrun $DISTRIBUTED_ARGS ringmo_pretrain_pp.py \
    $RINGMO_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
