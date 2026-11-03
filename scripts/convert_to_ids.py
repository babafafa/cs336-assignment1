import numpy as np
from cs336_basics.tokenizer import Tokenizer
import time
t = Tokenizer.from_files("tinystories_vocab.pkl", "tinystories_merges.pkl", special_tokens=["<|endoftext|>"])

def encode_file(tokenizer, input_path, output_path):
    print(f"开始编码 {input_path} ...")
    
    all_ids = []
    with open(input_path, "r", encoding="utf-8") as f:

        for token_id in tokenizer.encode_iterable(f):
            all_ids.append(token_id)
            
    # 转成 uint16 类型的 numpy 数组
    ids_array = np.array(all_ids, dtype=np.uint16)
    
    # 保存到硬盘
    np.save(output_path, ids_array)
    print(f"保存成功！文件大小: {ids_array.nbytes / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    start_time = time.time()
    encode_file(t, "../data/TinyStoriesV2-GPT4-train.txt", "tinystories_train_ids.npy")
    end_time1 = time.time()
    encode_file(t, "../data/TinyStoriesV2-GPT4-valid.txt", "tinystories_valid_ids.npy")
    end_time2= time.time()
    print(f"Train 编码耗时: {end_time1 - start_time:.2f} 秒")
    print(f"Valid 编码耗时: {end_time2 - end_time1:.2f} 秒")
    print(f"总耗时: {end_time2 - start_time:.2f} 秒")