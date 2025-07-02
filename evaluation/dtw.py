import pandas as pd
import ast
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import csv
import os
import numpy as np
def parse_minmax_list(column):
    """
    将字符串形式的列表转为真正的列表
    """
    return [ast.literal_eval(item) for item in column]

def compute_dtw_distances(file1, file2, output_path="dtw_distances.csv"):
    # 读取两个CSV文件
    df1 = pd.read_csv(file1, nrows=100)
    df2 = pd.read_csv(file2, nrows=100)

    # 解析 MinMax 列为真正的列表
    minmax_list1 = parse_minmax_list(df1["minmax"])
    minmax_list2 = parse_minmax_list(df2["minmax"])

    # 比较每一对序列（可根据需求调整匹配逻辑）
    results = []
    normalized_result=[]

    for i in range(100):
    #for i in range(min(len(minmax_list1), len(minmax_list2))):
        seq1=minmax_list1[i]
        seq2=minmax_list2[i]
        #print("1 seq1_clean",seq1)
        #print("1 seq2_clean",seq2)
       
        seq1_clean = [[x/100] for x in seq1 if x is not None]
        seq2_clean = [[x/100] for x in seq2 if x is not None]
        # print("2 seq1_clean",seq1_clean[:5])
        # print("2 seq2_clean",seq2_clean[:5])
        # print("type of seq1[0]:", type(seq1_clean[0]))
        # seq1_clean = np.ravel(seq1_clean)
        # seq2_clean = np.ravel(seq2_clean)
        seq1_clean = np.array(seq1_clean)  # 或 np.ravel(seq1_clean)
        seq2_clean =np.array(seq2_clean)
            
        distance, _ = fastdtw(seq1_clean, seq2_clean, dist=euclidean)
        normalized_distance = distance / max(len(seq1_clean), len(seq2_clean))
        #print("normalized_distance ",normalized_distance)
        #exit()
        #distance/=len(seq2_clean)
        #print("distance",distance)
        normalized_result.append(normalized_distance)
        results.append(["nature", "codontransform base", distance,normalized_distance])

    # 保存结果
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model1", "model2", "dtw_distance","normalized_distance"])
        writer.writerows(results)

    print(f"DTW distances saved to {output_path}")
    #exit()
