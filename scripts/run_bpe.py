"""
BPE 训练 + 存盘 + roundtrip 自检 + 编码 train/valid 为 npy + 统计。
用法：
    uv run python scripts/run_bpe.py
"""
import os
import pickle
import time
import random

import numpy as np

from cs336_basics.tokenizer import Tokenizer, TokenizerTrainer

# ---------------- 路径与超参（按需改） ----------------
TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VALID_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
OUT_DIR = "output"
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]
PROCESSOR_NUM = 8

VOCAB_FILE = os.path.join(OUT_DIR, "vocab.pkl")
MERGES_FILE = os.path.join(OUT_DIR, "merges.pkl")
TRAIN_NPY = os.path.join(OUT_DIR, "train.npy")
VALID_NPY = os.path.join(OUT_DIR, "valid.npy")


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# 一、训练 BPE
# ======================================================================
def train_bpe():
    print("=" * 60)
    print("[1/5] 训练 BPE")
    t0 = time.time()
    trainer = TokenizerTrainer(
        file_path=TRAIN_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        processor_num=PROCESSOR_NUM,
    )
    vocab, merges = trainer.train_bpe_pool()
    print(f"  耗时 {time.time() - t0:.1f}s")
    print(f"  vocab size = {len(vocab)}, merges = {len(merges)}")
    # 断言：词表尺寸
    assert len(vocab) == VOCAB_SIZE, (
        f"vocab 尺寸 {len(vocab)} != {VOCAB_SIZE}，"
        f"可能 merges 没跑够或 special token 分配异常"
    )
    return vocab, merges


# ======================================================================
# 二、存盘
# ======================================================================
def save_artifacts(vocab, merges):
    print("=" * 60)
    print("[2/5] 存盘")
    ensure_out_dir()
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump(vocab, f)
    with open(MERGES_FILE, "wb") as f:
        pickle.dump(merges, f)

    vocab_size_kb = os.path.getsize(VOCAB_FILE) / 1024
    merges_size_kb = os.path.getsize(MERGES_FILE) / 1024
    print(f"  {VOCAB_FILE}: {vocab_size_kb:.1f} KB")
    print(f"  {MERGES_FILE}: {merges_size_kb:.1f} KB")

    # 验收：vocab 几百 KB、merges 几十 KB
    assert vocab_size_kb > 50, (
        f"vocab.pkl 只有 {vocab_size_kb:.1f} KB，太小——是不是没存进去？"
    )
    assert merges_size_kb > 5, (
        f"merges.pkl 只有 {merges_size_kb:.1f} KB，太小——是不是没存进去？"
    )
    return vocab_size_kb, merges_size_kb


# ======================================================================
# 三、roundtrip 自检
# ======================================================================
def roundtrip_check():
    print("=" * 60)
    print("[3/5] roundtrip 自检")

    # 从文件重新 load，验证 from_files 路径正确
    tok = Tokenizer.from_files(VOCAB_FILE, MERGES_FILE, SPECIAL_TOKENS)

    # 抽 5 段原文，覆盖：换行、special token、中文/emoji、纯空白
    samples = [
        "Hello world.\nThis is a test.",
        "First doc.<|endoftext|>Second doc.<|endoftext|>",
        "中文测试：你好，世界！🚀 emoji 也要覆盖。",
        "    \n\n\n\t\t   ",              # 纯空白
        "Mixed 中文 and English, with\nnewlines<|endoftext|>and spaces.",
    ]

    for i, text in enumerate(samples):
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, (
            f"roundtrip 失败 [sample {i}]\n"
            f"  原文: {text!r}\n"
            f"  解码: {decoded!r}\n"
            f"  ids:  {ids[:20]}..."
        )
        print(f"  sample {i}: {len(ids)} tokens OK")

    print("  roundtrip 全部相等 ✓")


# ======================================================================
# 四、编码 train / valid 为 npy
# ======================================================================
def encode_to_npy(tok, path, out_path, name):
    print("=" * 60)
    print(f"[4/5] 编码 {name}: {path}")
    t0 = time.time()
    total = 0
    chunks = []
    for arr in tok.iter_chunks_parallel(path, n_workers=PROCESSOR_NUM):
        # array('I') -> np.uint16（确保 max < 65536）
        a = np.frombuffer(arr, dtype=np.uint32)
        if a.size and a.max() >= 65536:
            raise ValueError(f"{name} 有 id >= 65536，词表超过 uint16 范围")
        chunks.append(a.astype(np.uint16))
        total += a.size
    ids = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    np.save(out_path, ids)
    elapsed = time.time() - t0

    # 验收：一维、uint16、max < 10000
    assert ids.ndim == 1, f"{name} shape 不是一维：{ids.shape}"
    assert ids.dtype == np.uint16, f"{name} dtype 不是 uint16：{ids.dtype}"
    assert ids.size == 0 or int(ids.max()) < VOCAB_SIZE, (
        f"{name} max id = {ids.max()} >= {VOCAB_SIZE}，有越界 id"
    )

    file_size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  shape={ids.shape}, dtype={ids.dtype}, max={ids.max() if ids.size else 0}")
    print(f"  耗时 {elapsed:.1f}s, 文件 {file_size_mb:.1f} MB")
    return ids, elapsed, file_size_mb


# ======================================================================
# 五、打印统计
# ======================================================================
def print_stats(name, ids, raw_bytes, elapsed, file_size_mb):
    token_count = len(ids)
    compression = raw_bytes / token_count if token_count else 0
    print("=" * 60)
    print(f"[5/5] {name} 统计")
    print(f"  token 总数 : {token_count:,}")
    print(f"  原始字节数 : {raw_bytes:,}")
    print(f"  压缩率     : {compression:.3f} bytes/token")
    print(f"  编码耗时   : {elapsed:.1f}s")
    print(f"  npy 大小   : {file_size_mb:.1f} MB")

    # 验收：压缩率 3.0~3.8（TinyStories + 10k vocab 的经验区间）
    if compression < 2.5:
        print(f"  ⚠️ 压缩率 {compression:.2f} < 2.5：分词器没学到位，merge 可能有问题")
    elif compression > 4.5:
        print(f"  ⚠️ 压缩率 {compression:.2f} > 4.5：超过 GPT-2，肯定哪里错了")
    else:
        print(f"  ✓ 压缩率在健康区间 [3.0, 3.8] 附近")


# ======================================================================
# main
# ======================================================================
def main():
    # 1. 训练
    vocab, merges = train_bpe()

    # 2. 存盘
    save_artifacts(vocab, merges)

    # 3. roundtrip 自检
    roundtrip_check()

    # 4. 编码
    tok = Tokenizer.from_files(VOCAB_FILE, MERGES_FILE, SPECIAL_TOKENS)

    train_raw_bytes = os.path.getsize(TRAIN_PATH)
    train_ids, train_elapsed, train_file_mb = encode_to_npy(
        tok, TRAIN_PATH, TRAIN_NPY, "train")

    valid_raw_bytes = os.path.getsize(VALID_PATH)
    valid_ids, valid_elapsed, valid_file_mb = encode_to_npy(
        tok, VALID_PATH, VALID_NPY, "valid")

    # 5. 统计
    print_stats("train", train_ids, train_raw_bytes, train_elapsed, train_file_mb)
    print_stats("valid", valid_ids, valid_raw_bytes, valid_elapsed, valid_file_mb)

    print("=" * 60)
    print("全部完成 ✓")


if __name__ == "__main__":
    main()