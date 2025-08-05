
from functools import partial
import torch
import mindspeed.megatron_adaptor
from torch import nn
from megatron.core import parallel_state
from megatron.training import pretrain ,get_args, get_timers, print_rank_0
from megatron.core.enums import ModelType
from pytorch_for_ringmo import SwinTransformerForRingMo
from build_dataset_replace import create_pretrain_dataset
from megatron.core import mpu

# 尽量均分模型到每个stage
def get_pp_args(pipeline_size = 4):
    pp_args = {}
    if pipeline_size == 4:
        pp_args = {
            'depths': [2, 2, 12, 2],
            'num_heads': [2, 4, 8, 16],
            'num_local_experts': [16, 8, 4, 2],
            # 'moe_blocks': [[0,1], [0], [0], [0]],
            'blocks_pp_stages': [6, 6, 4, 2],
        }
    if pipeline_size == 8:
        # SwinV2-G 的block配置
        pp_args = {
            'depths': [2, 2, 42, 4],
            'num_heads': [4, 8, 16, 32],
            'num_local_experts': [64, 64, 32, 16],
            # 'moe_blocks': [[0,1], [0,1], list(range(42)), [0,1,2,3]],
            'blocks_pp_stages': [9, 6, 7, 6, 6, 7, 6, 3],
        }

    return pp_args

class DecoderWrapper(nn.Module):
    def __init__(self, conv, pixelshuffle):
        super().__init__()
        self.layers = nn.ModuleList([conv, pixelshuffle])  # 关键：提供 .layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def _masked_mean( loss, mask):
    """带掩码的加权平均"""
    # loss: [B,C,H,W], mask: [B,1,H,W]
    weighted_loss = torch.mul(loss, mask)
    sum_loss = torch.sum(weighted_loss)
    sum_mask = torch.sum(mask) + 1e-5  # 防止除零
    return sum_loss / sum_mask / 3 # in_chans

def ringmo_loss(x_rec, x_ori, mask, lbp=None, lbp_rec=None,):
    """ringmo loss"""
    # 计算原始重建损失
    loss_ori_recon = torch.abs(torch.sub(x_ori , x_rec))  # [B,C,H,W]
    loss_ori_mask = _masked_mean(loss_ori_recon, mask)
    
    # 计算LBP重建损失（如果启用）
    loss_lbp_mask = 0.
    return loss_ori_mask + loss_lbp_mask, {'recovery_loss':loss_ori_mask}

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    assert parallel_state.get_tensor_model_parallel_world_size() == 1 
    
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_args = get_pp_args(pipeline_size)

    # num of STBlocks of each BasicLayer when no pipeline parallel
    depths = pipeline_args['depths']
    # num_heads of each BasicLayer when no pipeline parallel
    num_heads = pipeline_args['num_heads']
    # STBlocks number in this pp_stage
    blocks_pp_stages = pipeline_args['blocks_pp_stages']
    # 使用moe的blocks在所在层的id.
    if 'moe_blocks' in pipeline_args:
        moe_blocks = pipeline_args['moe_blocks']
    else:
        moe_blocks = []
    # 每级的专家数
    num_local_experts = pipeline_args['num_local_experts']

    # 完整模型配置
    stage_model_args = {
        'img_size': 192,
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 0,
        'embed_dim': 32,
        'window_size': 6,
        'mlp_ratio': 4,
        'ape': False,
        'patch_norm': True,
        'depths': depths,
        'num_heads': num_heads,
        'pipeline_model_parallel_size': pipeline_size,
        'pipeline_dtype': torch.float16,
        # moe args
        'moe_blocks': moe_blocks,
        'num_local_experts': num_local_experts,
        'top_value': 2,
        # pp args
        'pre_process': pre_process,
        'post_process': post_process,
        'blocks_pp_stages': blocks_pp_stages,
    }

    # log = '------------------------\n'
    # import torch.distributed as dist
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # log += f"Global rank: {rank}\n"
    # log += f"World size: {world_size}\n"
    # log += f"DP size: {mpu.get_data_parallel_world_size()}\n"
    # log += f"PP size: {mpu.get_pipeline_model_parallel_world_size()}\n"
    # log += f"my PP Rank: {mpu.get_pipeline_model_parallel_rank()}"
    # print(log)

    model = SwinTransformerForRingMo(**stage_model_args)
    
    # 处理输入预处理
    if not mpu.is_pipeline_first_stage():
        model.patch_embed = nn.Identity()
        model.pos_drop = nn.Identity()
    
    # 添加解码器组件
    if mpu.is_pipeline_last_stage():
        conv = nn.Conv2d(model.num_features, (32**2) * 3, kernel_size=1)
        pixelshuffle = nn.PixelShuffle(32)
        model.decoder = DecoderWrapper(conv, pixelshuffle)

    # pp_rank = mpu.get_pipeline_model_parallel_rank()
    # print(f"----------------------- pp rank {pp_rank} ------------------------")
    # print(model)
    return model.cuda()

# 前向传播函数
def forward_step(data_iterator, model):
    x ,mask_in = None , None
    args = get_args()

    timers = get_timers() # Get the batch.
    
    timers('batch-generator', log_level=2).start()
    data = next(data_iterator)
    timers('batch-generator').stop()

    if parallel_state.is_pipeline_first_stage():
        x = data["image"].cuda()
        mask_in = data["mask"].cuda().float()
        x_output  = model((x , mask_in)) 
    elif not parallel_state.is_pipeline_last_stage():
        x_output  = model(None) 
    else :
        x_output  = model(None) 

    return x_output, partial(ringmo_loss, x_ori=data['image'].cuda(), mask=data["mask"].cuda())

# 数据集提供函数
def train_valid_test_dataset_provider(train_val_test_num_samples):
    train_ds = create_pretrain_dataset(get_args())
    # train_ds = iter(train_ds)
    return train_ds, None, None

# 主函数
if __name__ == "__main__":

    pretrain(
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=model_provider,
        forward_step_func=forward_step,
        model_type=ModelType.encoder_or_decoder,
    )
    
