import time
import pickle
import numpy as np
from array import array
from cs336_basics.tokenizer import Tokenizer

# ==========================================
# 1. 加载你的 Tokenizer
# ==========================================
# ts_tokenizer = Tokenizer.from_files("tinystories_vocab.pkl", "tinystories_merges.pkl", ["<|endoftext|>"])
owt_tokenizer = Tokenizer.from_files("/root/cs336-assignment1/data/processed/owt_vocab.pkl", "/root/cs336-assignment1/data/processed/owt_merges.pkl", ["<|endoftext|>"])

# # 读取 10 个文档的采样文本
# with open("sampled_tinystories.txt", "r", encoding="utf-8") as f:
#     ts_text = f.read()
# with open("sampled_owt.txt", "r", encoding="utf-8") as f:
#     owt_text = f.read()


# # ==========================================
# # (a) 计算压缩率
# # ==========================================
# def calc_compression_ratio(text, tokenizer):
#     byte_count = len(text.encode('utf-8'))
#     ids = tokenizer.encode(text)
#     token_count = len(ids)
#     return byte_count / token_count

# print("--- 问题 (a) ---")
# print(f"TinyStories Tokenizer 对 TS 文本压缩率: {calc_compression_ratio(ts_text, ts_tokenizer):.2f} bytes/token")
# print(f"OWT Tokenizer 对 OWT 文本压缩率: {calc_compression_ratio(owt_text, owt_tokenizer):.2f} bytes/token")


# # ==========================================
# # (b) 交叉测试
# # ==========================================
# print("--- 问题 (b) ---")
# print(f"TinyStories Tokenizer 对 OWT 文本压缩率: {calc_compression_ratio(owt_text, ts_tokenizer):.2f} bytes/token")


# # ==========================================
# # (c) 吞吐量预估
# # ==========================================
# def calc_throughput(text, tokenizer):
#     byte_count = len(text.encode('utf-8'))
#     start = time.time()
#     tokenizer.encode(text)
#     end = time.time()
#     return byte_count / (end - start)

# print("--- 问题 (c) ---")
# throughput = calc_throughput(ts_text, ts_tokenizer) # 用一小段文本测试速度
# print(f"吞吐量: {throughput / 1024 / 1024:.2f} MB/s")
# pile_gb = 825
# pile_bytes = pile_gb * 1024 * 1024 * 1024
# estimated_seconds = pile_bytes / throughput
# print(f"预估处理 Pile 数据集需要: {estimated_seconds / 3600 / 24:.1f} 天")


# ==========================================
# (d) 编码整个数据集到 uint16 数组
# ==========================================
def encode_to_uint16(tokenizer, input_path, output_path):
    print(f"编码 {input_path} ...")
    all_ids = array('H')  # 'H' 就是 uint16
    
    with open(input_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            all_ids.append(token_id)
            
    # 转成 numpy 并保存
    ids_array = np.frombuffer(all_ids, dtype=np.uint16)
    np.save(output_path, ids_array)
    print(f"保存至 {output_path}，大小: {ids_array.nbytes / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # encode_to_uint16(ts_tokenizer, "/root/assignment1/data/processed/TinyStoriesV2-GPT4-train.txt", "ts_train_ids.npy")
    # encode_to_uint16(ts_tokenizer,"/root/assignment1/data/processed/TinyStoriesV2-GPT4-valid.txt", "ts_valid_ids.npy")
    encode_to_uint16(owt_tokenizer, "/root/assignment1/data/processed/owt_train.txt", "owt_train_ids.npy")
    encode_to_uint16(owt_tokenizer, "/root/assignment1/data/processed/owt_valid.txt", "owt_valid_ids.npy")