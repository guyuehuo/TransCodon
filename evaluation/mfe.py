import pandas as pd
import RNA
import os
import multiprocessing

# **定义并行计算 MFE 的函数**
def predict_mfe(sequence):
    """使用 ViennaRNA 计算 RNA 二级结构的 MFE"""
    structure, mfe = RNA.fold(sequence)  # 计算 MFE
    return mfe

# **定义一个包装函数，用于多进程调用**
def compute_mfe_parallel(sequences, num_workers=None):
    """使用多进程计算多个 RNA 序列的 MFE"""
    if num_workers is None:
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
        num_workers =32  # 默认使用 80% 的 CPU 核心
        print("num_workers",num_workers)
    with multiprocessing.Pool(num_workers) as pool:
        mfe_values = pool.map(predict_mfe, sequences)  # 并行计算 MFE
    return mfe_values
