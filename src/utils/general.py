import os

import torch


def get_num_gpus():
    # Respect CUDA_VISIBLE_DEVICES if set, otherwise count all
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        n = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        n = torch.cuda.device_count()

    # vLLM performs best when TP is a power of 2 (1, 2, 4, 8)
    # This logic caps it at the highest power of 2 available
    return 1 << (n.bit_length() - 1) if n > 0 else 1
