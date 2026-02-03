import os
import random

import numpy as np
import torch


def enforce_reproducibility(seed=42):
    # 1. Python & Numpy seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # 3. CuDNN backend settings
    # This forces deterministic algorithms and disables the "benchmarker"
    # which usually picks the fastest (and often non-deterministic) algo.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 4. CUDA environment variables
    # Required for certain versions of CUDA to ensure deterministic behavior
    # in operations like atomicAdd.
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 5. Force PyTorch to throw an error if a non-deterministic op is used
    # This is the "strict mode" to ensure 100% compliance.
    # torch.use_deterministic_algorithms(True)

    print(f"✅ Reproducibility enforced with seed: {seed}")
