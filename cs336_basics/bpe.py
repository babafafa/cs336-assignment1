# import regex as re
# from collections import defaultdict, Counter
# import multiprocessing as mp
# from cs336_basics.pretokenization_example import find_chunk_boundaries

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# def process_chunk(args):

#     file_path, start, end, special_tokens = args
#     local_counts = Counter()
    
#     with open(file_path, "rb") as f:
#         f.seek(start)
#         chunk_bytes = f.read(end - start)
#         chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
        
#     if special_tokens:
#         escaped = [re.escape(t) for t in special_tokens]
#         pattern = "|".join(escaped)
#         sub_chunks = re.split(pattern, chunk_text)
#     else:
#         sub_chunks = [chunk_text]

#     for chunk in sub_chunks:
#         for match in re.finditer(PAT, chunk):
#             b_word = match.group().encode('utf-8')
#             token_tuple = tuple(b_word[i:i+1] for i in range(len(b_word)))
#             local_counts[token_tuple] += 1
            
#     return local_counts
    
# def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
#     new_word = []
#     i = 0
#     while i < len(word):
#         if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
#             new_word.append(pair[0]+pair[1])
#             i += 2
#         else:
#             new_word.append(word[i])
#             i += 1
#     return tuple(new_word)

# def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
#     dict[int, bytes], list[tuple[bytes, bytes]]]:
#     assert vocab_size >= 256 + len(special_tokens)

#     # --- 初始化词表 ---
#     vocab = {i: bytes([i]) for i in range(256)}
#     current_id = 256
#     for t in special_tokens:
#         vocab[current_id] = t.encode('utf-8')
#         current_id += 1
#     word_counts = Counter()
#     chunk_multiplier=8
#     cpu_count = 16
#     with mp.Pool(cpu_count) as pool:
#         with open(input_path, "rb") as f:
#             # 只传第一个特殊 token 作为切分边界
#             split_token = special_tokens[0].encode('utf-8') if special_tokens else b"<|endoftext|>"
#             num_chunks = cpu_count*chunk_multiplier
#             boundaries = find_chunk_boundaries(f, num_chunks, split_token)
            
#         tasks = []
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             tasks.append((input_path, start, end, special_tokens))
            
#         for local_counts in pool.imap_unordered(process_chunk, tasks):
#             word_counts.update(local_counts)

#     # --- 构建初始 Pair 频次与反向索引 ---
#     pair_counts = defaultdict(int)
#     pair_to_words = defaultdict(set)
#     for word, freq in word_counts.items():
#         for i in range(len(word) - 1):
#             pair = (word[i], word[i + 1])
#             pair_counts[pair] += freq
#             pair_to_words[pair].add(word)

#     # --- 核心合并循环 ---
#     merges = []
    
#     import gc
#     gc.disable() 
#     try:   
#         while len(vocab) < vocab_size:
#             if not pair_counts:
#                 break
    
#             # 填 3：找到最高频的 pair（频次相同选字典序大的）
#             best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
#             merges.append(best_pair)
#             new_token = best_pair[0] + best_pair[1]
#             vocab[current_id] = new_token
#             current_id += 1
    
#             affected_words = list(pair_to_words[best_pair])
    
#             for old_word in affected_words:
#                 if old_word not in word_counts:
#                     continue
#                 freq = word_counts[old_word]
    
#                 # 拆旧：减去频次
#                 for i in range(len(old_word) - 1):
#                     old_p = (old_word[i], old_word[i + 1])
#                     pair_counts[old_p] -= freq
#                     if pair_counts[old_p] == 0:
#                         del pair_counts[old_p]
#                     pair_to_words[old_p].discard(old_word)
    
#                 # 生成新词
#                 new_word = merge_word(old_word, best_pair)
    
#                 # 装新：加上频次
#                 for i in range(len(new_word) - 1):
#                     new_p = (new_word[i], new_word[i + 1])
#                     pair_counts[new_p] += freq
#                     pair_to_words[new_p].add(new_word)
    
#                 # 更新主词频表
#                 del word_counts[old_word]
#                 word_counts[new_word] += freq
#     finally:
#         gc.enable()   
#         gc.collect()  

#     return vocab, merges

import gc
import multiprocessing
import os
import traceback
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re  # 预分词必须用 regex，支持 \p{L} 等 Unicode 属性
from tqdm import tqdm

# GPT-2 风格的预分词正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class TokenizerTrainer:
    def __init__(self, file_path, vocab_size, special_tokens, processor_num):
        """
        :param file_path: 训练分词器的文本文件路径
        :param vocab_size: 最终词表尺寸（含初始256字节词表、合并项和特殊token）
        :param special_tokens: 特殊token列表，例如 ["<|endoftext|>"]
        :param processor_num: 进程数量（train_bpe_pool 中使用）
        """
        self.file_path = file_path
        self.vocab_size = vocab_size if vocab_size else 512
        self.special_tokens = special_tokens if special_tokens else []
        self.processor_num = processor_num
        # 数据分块数 = 进程数 * 该倍数。分更多小块可避免整份训练集常驻内存导致 OOM
        self.chunk_multiplier = 10

        # 初始化词表：256 个字节值 + 特殊 token（特殊 token 有固定 ID）
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(self.special_tokens):
            self.vocab[256 + i] = token.encode("utf-8")
        self.merges = []

        # pre_tokens: {pre_token_index: [bytes_tuple, count]}
        self.pre_tokens = {}
        # pre_token_to_index: {bytes_tuple: pre_token_index}
        self.pre_token_to_index = {}
        # bp_freq: {(token_id1, token_id2): [count, [pre_token_index, ...]]}
        # 其中 token_id 既可以是字节值，也可以是后续合并产生的新 token ID
        self.bp_freq = {}

    def train_bpe(self):
        """
        Train byte pair encoding. 单进程版本。
        返回 (vocab, merges)：
          - vocab: dict[int, bytes]，token ID -> 字节串
          - merges: list[tuple[bytes, bytes]]，按创建顺序排列的合并项
        """
        # 预分词前按特殊 token 分割语料，保证不会在文档边界（特殊 token 处）发生合并
        concat_special_tokens = "|".join(re.escape(t) for t in self.special_tokens) if self.special_tokens else None

        with open(self.file_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.processor_num, b"<|endoftext|>")

            i = 0
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                segments = re.split(concat_special_tokens, chunk) if concat_special_tokens else [chunk]
                for seg in segments:
                    # 使用 finditer 避免一次性存储全部预分词结果
                    for m in re.finditer(PAT, seg):
                        pre_token = tuple(m.group().encode("utf-8"))
                        if pre_token not in self.pre_token_to_index:
                            self.pre_token_to_index[pre_token] = i
                            self.pre_tokens[i] = [pre_token, 1]
                            i += 1
                        else:
                            self.pre_tokens[self.pre_token_to_index[pre_token]][1] += 1

        self.init_bp_freq()
        for _ in range(self.vocab_size - len(self.vocab)):
            self.merge_bpe()
        return self.vocab, self.merges

    def train_bpe_pool(self):
        """
        使用内存优化的多进程池管理分词器训练。
        预分词用进程池并行，合并阶段仍为单进程（BPE 合并无法并行化）。
        """
        # 1. 只计算文件分块位置（不读取内容），分块数远大于进程数，
        #    每块约 30-50MB，imap_unordered 自动调度，同一时间只有
        #    processor_num 个进程在读文件，降低峰值内存。
        num_chunks = self.processor_num * self.chunk_multiplier
        with open(self.file_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

        pool = multiprocessing.Pool(processes=self.processor_num)

        # 2. 只传文件路径和偏移量，不传大字符串，避免进程间序列化开销
        tasks = [(self.file_path, start, end, self.special_tokens)
                 for start, end in zip(boundaries[:-1], boundaries[1:])]

        pbar = tqdm(total=len(tasks), desc="Pretokenizing")

        # 3. 大量字典更新时禁用 gc 以提速
        gc.disable()
        self.token_freqs = {}
        try:
            for worker_result in pool.imap_unordered(TokenizerTrainer.pretokenize_file_offset, tasks):
                if worker_result:
                    self._incremental_merge(worker_result)
                pbar.update(1)
        finally:
            pbar.close()
            pool.close()
            pool.join()

        self.pre_tokens = {i: [token, count]
                           for i, (token, count) in enumerate(self.token_freqs.items())}
        del self.token_freqs

        # 5. 恢复 gc，确保内存回收正常
        gc.enable()
        gc.collect()

        # 6. 初始化 byte pair 计数并迭代合并
        self.init_bp_freq()

        pbar = tqdm(total=self.vocab_size - len(self.vocab), desc="Merge byte pairs")
        for _ in range(self.vocab_size - len(self.vocab)):
            self.merge_bpe()
            pbar.update(1)
        pbar.close()

        return self.vocab, self.merges

    @staticmethod
    def pretokenize_file_offset(args):
        """子进程入口：自行读取文本块并预分词，返回 {token_tuple: count}。"""
        file_path, start, end, special_tokens_list = args
        try:
            with open(file_path, "rb") as f:
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

            pre_tokens = {}
            concat_special_tokens = "|".join(re.escape(t) for t in special_tokens_list) if special_tokens_list else None
            segments = re.split(concat_special_tokens, chunk_str) if concat_special_tokens else [chunk_str]
            for seg in segments:
                for m in re.finditer(PAT, seg):
                    pre_token = tuple(m.group().encode("utf-8"))
                    pre_tokens[pre_token] = pre_tokens.get(pre_token, 0) + 1
            return pre_tokens

        except Exception as e:
            print(f"Error in process {os.getpid()}: {e}")
            print(traceback.format_exc())
            return None

    def _incremental_merge(self, worker_result):
        """将单个子进程的结果合并到主进程的 token_freqs。"""
        for token_bytes, count in worker_result.items():
            self.token_freqs[token_bytes] = self.token_freqs.get(token_bytes, 0) + count

    def init_bp_freq(self):
        """遍历全部 pretoken，统计相邻 byte pair（token pair）的数量及出现位置。"""
        for pre_token_index, (pre_token, pre_count) in self.pre_tokens.items():
            all_bps = list(zip(pre_token[:-1], pre_token[1:]))
            for bp in all_bps:
                if bp not in self.bp_freq:
                    self.bp_freq[bp] = [0, []]
                self.bp_freq[bp][0] += pre_count
            # 记录每个 pair 出现在哪些 pretoken 中
            for bp in set(all_bps):
                self.bp_freq[bp][1].append(pre_token_index)

    def merge_bpe(self):
        """找到频数最高的 pair 合并，并更新词表与 merges。"""
        # 例如 max([("A","B"),("A","C"),("B","ZZ"),("BA","A")]) == ("BA","A")
        pair = max(self.bp_freq,
                   key=lambda x: (self.bp_freq[x][0], self.vocab[x[0]], self.vocab[x[1]]))
        index1, index2 = pair
        new_index = len(self.vocab)
        # 新 token = 两个 token 的字节串拼接
        self.vocab[new_index] = self.vocab[index1] + self.vocab[index2]
        # merges 按创建顺序记录 (bytes, bytes)
        self.merges.append((self.vocab[index1], self.vocab[index2]))
        self.merge(pair, new_index)

    def merge(self, pair, new_index):
        """
        合并 pair 并增量更新 bp_freq。

        每次合并只会改变"合并位置附近重叠区域"的 pair 计数：
          - 消失的 pair：(前一个 token, pair[0]) 和 (pair[1], 后一个 token)
          - 新增的 pair：(前一个 token, new_token) 和 (new_token, 后一个 token)
          因此只需遍历涉及该 pair 的 pretoken，无需遍历全部语料。
        """
        have_pair_tokens = self.bp_freq[pair][1]
        for have_pair_token_index in have_pair_tokens:
            bytestr = self.pre_tokens[have_pair_token_index][0]
            new_pre_token = bytestr
            pre_count = self.pre_tokens[have_pair_token_index][1]

            i = 0
            while i < len(new_pre_token) - 1:
                slid_window = new_pre_token[i:i + 2]
                if pair == slid_window:
                    # 目标 pair 前面还有 token：原 (prev, pair[0]) 消失，新增 (prev, new)
                    if i > 0:
                        new_pair = (new_pre_token[i - 1], new_index)
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count, [have_pair_token_index]]
                        else:
                            self.bp_freq[new_pair][0] += pre_count
                            if not self.bp_freq[new_pair][1] or self.bp_freq[new_pair][1][-1] != have_pair_token_index:
                                self.bp_freq[new_pair][1].append(have_pair_token_index)

                        overlap_pair = (new_pre_token[i - 1], pair[0])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <= 0:
                            self.bp_freq.pop(overlap_pair)

                    # 目标 pair 后面还有 token：原 (pair[1], next) 消失，新增 (new, next)
                    if i + 2 < len(new_pre_token):
                        new_pair = (new_index, new_pre_token[i + 2])
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count, [have_pair_token_index]]
                        else:
                            self.bp_freq[new_pair][0] += pre_count
                            if not self.bp_freq[new_pair][1] or self.bp_freq[new_pair][1][-1] != have_pair_token_index:
                                self.bp_freq[new_pair][1].append(have_pair_token_index)

                        overlap_pair = (pair[1], new_pre_token[i + 2])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <= 0:
                            self.bp_freq.pop(overlap_pair)

                    # 目标 pair 自身计数减少，归零则移除
                    self.bp_freq[pair][0] -= pre_count
                    if self.bp_freq[pair][0] <= 0:
                        self.bp_freq.pop(pair)

                    # 就地合并：替换为新 token ID
                    new_pre_token = new_pre_token[:i] + (new_index,) + new_pre_token[i + 2:]

                i += 1

            # 更新 pretoken 的字节序列
            self.pre_tokens[have_pair_token_index][0] = new_pre_token




