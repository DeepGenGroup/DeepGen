from pytorch_for_ringmo import *



def runRingmo() :
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
        'pipeline_dtype': torch.float32,
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

