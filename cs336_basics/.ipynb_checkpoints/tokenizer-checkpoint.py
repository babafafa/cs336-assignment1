# import regex as re
# from typing import Iterable, Iterator
# import pickle

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# def apply_merges(word_tokens, merge_ranks):
#     while len(word_tokens) > 1:
#         best_rank = None
#         best_i = -1
#         for i in range(len(word_tokens) - 1):
#             r = merge_ranks.get((word_tokens[i], word_tokens[i+1]))
#             if r is not None and (best_rank is None or r < best_rank):
#                 best_rank = r
#                 best_i = i
#         if best_i == -1:
#             break
#         word_tokens[best_i:best_i+2] = [word_tokens[best_i] + word_tokens[best_i+1]]
#     return word_tokens
    
# class Tokenizer:
#     def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens=special_tokens        
#         # 1. 建立反向映射表
#         self.vocab_to_id = {v: k for k, v in self.vocab.items()}
#         self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
#         # 2. 处理 special_tokens
#         if self.special_tokens:
#             # 去重 + 按长度降序，长 token 优先匹配
#             ordered = sorted(set(self.special_tokens), key=len, reverse=True)
#             escaped = [re.escape(t) for t in ordered]
#             self.special_pattern = "(" + "|".join(escaped) + ")"
            
#             # 遍历每一个特殊 token，分配 ID
#             for t in self.special_tokens:
#                 b_t = t.encode('utf-8')  # 转成 bytes
#                 if b_t not in self.vocab_to_id:  # 判断它是否已经在字典里
#                     new_id = len(self.vocab)     # 分配最新的 ID
#                     self.vocab[new_id] = b_t
#                     self.vocab_to_id[b_t] = new_id
#         else:
#             self.special_pattern = ""
        
#     @classmethod
#     def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
#         # 用 pickle 从文件里加载 vocab 和 merges
#         # 然后 return cls(vocab, merges, special_tokens)
#         with open(vocab_filepath, "rb") as f:
#             vocab = pickle.load(f)  
#         with open(merges_filepath, "rb") as f:
#             merges = pickle.load(f)    
#         return cls(vocab,merges,special_tokens)



#     def encode(self, text: str) -> list[int]:
#         ids = []
#         if self.special_pattern:
#             chunks = re.split(self.special_pattern, text)
#         else:
#             chunks = [text]
    
#         for chunk in chunks:
#             if not chunk:
#                 continue
#             if self.special_tokens and chunk in self.special_tokens:
#                 ids.append(self.vocab_to_id[chunk.encode('utf-8')])
#                 continue
#             for match in re.finditer(PAT, chunk):
#                 b_word = match.group().encode('utf-8')
#                 word_tokens = [b_word[i:i+1] for i in range(len(b_word))]
#                 word_tokens = apply_merges(word_tokens, self.merge_ranks)
#                 for token in word_tokens:
#                     ids.append(self.vocab_to_id[token])
#         return ids
        
#     def encode_iterable(self, iterable: Iterable[int]):
#         """
#         根据传入的可迭代文件对象 iterable，分块encode，保证内存高效性。
#         核心逻辑是分块读取时保证不会打碎特殊token或者pretoken。我的实现逻辑是在文本块结尾处向前取2倍最长token长度保存为end_chunk，检测end_chunk中是否存在special_token，如果存在截取chunk到specialtoken前，如果不存在则进行预分词，并抛弃最后一个pretoken（这样可以保证分块不会打碎pretoken）。
#         """
#         chunk_size = int(1 * 1024 ** 2)
#         PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#         # 初始化文件读取指针位置
#         file_pos = 0
#         while True:
#             chunk = iterable.read(chunk_size)
#             # 获取当前文件指针位置
#             pos = iterable.tell()

#             # 检查是否到达文件末尾，到末尾则编码结束
#             if not chunk:
#                 break
            
#             # 判断是否是最后一块（读取长度小于chunk_size说明是最后一块）
#             # 若是最后一块则可以跳过pretoken和specialtoken的打碎判断逻辑
#             is_last_chunk = len(chunk) < chunk_size
#             original_chunk = chunk
            
#             if not is_last_chunk:
#                 # 不是最后一块，需要截断最后一个可能不完整的token
#                 if self.special_tokens:
#                     concat_special_tokens = "|".join([re.escape(special_token) for special_token in self.special_tokens])
#                     end_chunk = chunk[-2 * len(max(self.special_tokens, key=len)) :]
#                     matches = list(re.finditer(concat_special_tokens, end_chunk))
#                     if len(matches) > 0:
#                         offset = len(chunk) - len(end_chunk)          # end_chunk 在 chunk 中的起点
#                         chunk = chunk[:offset + matches[0].span()[0]] 
#                     else:
#                         chunk = chunk[:-len(max(self.special_tokens, key=len))]
#                         pretokens = re.findall(PAT, chunk)
#                         if pretokens:
#                             endtoken_len = len(pretokens[-1])
#                             chunk = chunk[:-endtoken_len]
#                 else:
#                     pretokens = re.findall(PAT, chunk)
#                     if pretokens:
#                         endtoken_len = len(pretokens[-1])
#                         chunk = chunk[:-endtoken_len]
                
#                 # 如果截断后chunk为空，说明整个chunk可能是一个很长的token
#                 # 在这种情况下，使用原始chunk（不截断），否则会死循环
#                 if not chunk:
#                     chunk = original_chunk

#             chunk_len = len(chunk)
            
#             # 如果chunk为空，说明已经处理完所有内容（防御式编程，主要为了避免死循环）
#             if chunk_len == 0:
#                 break
#             # 获取字符串截断后对应的文件指针位置
#             iterable.seek(file_pos)
#             iterable.read(chunk_len)
#             file_pos = iterable.tell()
#             # print("file pos is", file_pos)
#             iterable.seek(file_pos)
#             token_ids = self.encode(chunk)
#             # 生成器，逐个输出token_id
#             for token_id in token_ids:
#                 yield token_id

                
#     def decode(self, ids: list[int]) -> str:
#         # 1. 收集所有的字节片段
#         byte_parts = []
#         for i in ids:
#             if i in self.vocab: 
#                 byte_parts.append(self.vocab[i])
#             else:
#                 byte_parts.append(b'') 
                
#         # 2. 把它们拼成一个完整的长字节串
#         b = b"".join(byte_parts)
        
#         # 3. 解码成字符串，遇到残缺字节用官方替换符
#         return b.decode("utf-8", errors="replace")

from typing import Iterable, Iterator
import pickle

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
        self.special_tokens = special_tokens if special_tokens else []
        assert vocab_size is not None and vocab_size >= 256 + len(self.special_tokens), \
            f"vocab_size={vocab_size} 太小，至少 256 + {len(self.special_tokens)}"
        self.vocab_size = vocab_size
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
        for pre_token_index, (pre_token, pre_count) in self.pre_tokens.items():
            all_bps = list(zip(pre_token[:-1], pre_token[1:]))
            for bp in all_bps:
                if bp not in self.bp_freq:
                    self.bp_freq[bp] = [0, set()]          # ← set
                self.bp_freq[bp][0] += pre_count
            for bp in set(all_bps):
                self.bp_freq[bp][1].add(pre_token_index)    # ← add

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
        # ① 快照 + 过滤死索引（同时天然去重，因为 [1] 已是 set）
        #    并顺手做"仍含 pair"检查：不含的直接丢弃
        alive = []
        for idx in self.bp_freq[pair][1]:
            tokens = self.pre_tokens[idx][0]
            # 快速判断 tokens 里是否还存在相邻的 (pair[0], pair[1])
            found = False
            for j in range(len(tokens) - 1):
                if tokens[j] == pair[0] and tokens[j + 1] == pair[1]:
                    found = True
                    break
            if found:
                alive.append(idx)

        # ② 立即清空：后面若产生 new_pair == pair 的登记，会重新 add 进来；
        #    而已处理的活索引处理完必不再含 pair，不该留在列表里
        self.bp_freq[pair][1].clear()

        # ③ 逐个处理活索引（逻辑同原版，只把 append 换成 add）
        for have_pair_token_index in alive:
            if pair not in self.bp_freq:      # 计数已归零，剩余活索引不再有贡献
                break
            bytestr = self.pre_tokens[have_pair_token_index][0]
            new_pre_token = bytestr
            pre_count = self.pre_tokens[have_pair_token_index][1]

            i = 0
            while i < len(new_pre_token) - 1:
                slid_window = new_pre_token[i:i + 2]
                if pair == slid_window:
                    # 左邻居
                    if i > 0:
                        new_pair = (new_pre_token[i - 1], new_index)
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count, {have_pair_token_index}]
                        else:
                            self.bp_freq[new_pair][0] += pre_count
                            self.bp_freq[new_pair][1].add(have_pair_token_index)

                        overlap_pair = (new_pre_token[i - 1], pair[0])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <= 0:
                            self.bp_freq.pop(overlap_pair, None)

                    # 右邻居
                    if i + 2 < len(new_pre_token):
                        new_pair = (new_index, new_pre_token[i + 2])
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count, {have_pair_token_index}]
                        else:
                            self.bp_freq[new_pair][0] += pre_count
                            self.bp_freq[new_pair][1].add(have_pair_token_index)

                        overlap_pair = (pair[1], new_pre_token[i + 2])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <= 0:
                            self.bp_freq.pop(overlap_pair, None)

                    # pair 自身计数减少（注意：此时 pair 的 [1] 已 clear，
                    # 若减到 0 直接 pop 整个 pair 即可）
                    self.bp_freq[pair][0] -= pre_count
                    if self.bp_freq[pair][0] <= 0:
                        self.bp_freq.pop(pair, None)

                    # 就地合并
                    new_pre_token = new_pre_token[:i] + (new_index,) + new_pre_token[i + 2:]

                i += 1

            self.pre_tokens[have_pair_token_index][0] = new_pre_token
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # 1. 建立反向映射表
        self.vocab_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        # 2. 处理 special_tokens
        if self.special_tokens:
            # 去重 + 按长度降序，长 token 优先匹配
            ordered = sorted(set(self.special_tokens), key=len, reverse=True)
            escaped = [re.escape(t) for t in ordered]
            self.special_pattern = "(" + "|".join(escaped) + ")"

            # 遍历每一个特殊 token，分配 ID
            for t in self.special_tokens:
                b_t = t.encode('utf-8')  # 转成 bytes
                if b_t not in self.vocab_to_id:  # 判断它是否已经在字典里
                    new_id = len(self.vocab)     # 分配最新的 ID
                    self.vocab[new_id] = b_t
                    self.vocab_to_id[b_t] = new_id
        else:
            self.special_pattern = ""

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _encode_with_cache(self, text: str, cache: dict) -> list[int]:
        """
        与 encode 等价，但共享跨 chunk 的 cache：
        cache: {pretoken_bytes: tuple(合并后的 token)}
        只对本块"cache 未命中"的 unique pretoken 调用 merge_pretokens_batch。
        """
        ids = []
        if self.special_pattern:
            chunks = re.split(self.special_pattern, text)
        else:
            chunks = [text]

        pre_tokens = {}          # 只收集 cache 未命中的 unique pretoken
        pre_tokens_seq = []      # 按出现顺序：bytes（普通）或 str（特殊）

        special_set = set(self.special_tokens) if self.special_tokens else set()

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_set:                 # 特殊 token，不参与 merge
                pre_tokens_seq.append(chunk)
                continue
            for m in re.finditer(PAT, chunk):
                b = m.group().encode('utf-8')
                pre_tokens_seq.append(b)
                if b not in cache and b not in pre_tokens:
                    pre_tokens[b] = [b[i:i+1] for i in range(len(b))]

        # 只 merge 未命中的
        if pre_tokens:
            merged = self.merge_pretokens_batch(pre_tokens)
            cache.update(merged)

        # 按顺序展开
        for item in pre_tokens_seq:
            if isinstance(item, str):                # 特殊 token
                ids.append(self.vocab_to_id[item.encode('utf-8')])
            else:                                    # 普通 pretoken（必在 cache 里）
                for tok in cache[item]:
                    ids.append(self.vocab_to_id[tok])
        return ids
    # ------------------------------------------------------------------
    # 新增：批量 merge，使用 bp_to_pretoken 反向索引
    # ------------------------------------------------------------------
    def merge_pretokens_batch(self, pre_tokens: dict[bytes, list[bytes]]) -> dict[bytes, tuple]:
        """
        对一批 unique pretoken 做 BPE merge。
        输入:  {pretoken_bytes: [单字节bytes, ...]}   (初始每个字节一个 token)
        输出:  {pretoken_bytes: (合并后的 token, ...)}
        思路:
          1) 建反向索引 bp_to_pretoken: {(b1, b2): {pretoken, ...}}
          2) 按 self.merges 训练顺序执行, 只处理包含当前 pair 的 pretoken
          3) 合并后产生的相邻对增量加入索引
        """
        bp_to_pretoken = {}   # {(b1, b2): {pretoken_bytes, ...}}

        # 1) 建反向索引
        for pre_token, tokens in pre_tokens.items():
            i = 0
            while i < len(tokens) - 1:
                bp = (tokens[i], tokens[i + 1])
                bp_to_pretoken.setdefault(bp, set()).add(pre_token)
                i += 1

        # 2) 按 merges 顺序执行
        for merge in self.merges:
            if merge not in bp_to_pretoken:
                continue
            # 复制一份再遍历, 防止合并中新产生的对恰好等于当前 merge 时修改 set 报错
            for pre_token in list(bp_to_pretoken[merge]):
                tokens = pre_tokens[pre_token]
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i + 1]) == merge:
                        tokens[i] = tokens[i] + tokens[i + 1]
                        tokens.pop(i + 1)
                        # 合并后新产生的相邻对增量加入索引
                        if i > 0:
                            bp_prev = (tokens[i - 1], tokens[i])
                            bp_to_pretoken.setdefault(bp_prev, set()).add(pre_token)
                        if i < len(tokens) - 1:
                            bp_next = (tokens[i], tokens[i + 1])
                            bp_to_pretoken.setdefault(bp_next, set()).add(pre_token)
                    else:
                        i += 1
                pre_tokens[pre_token] = tokens

        # 3) 转成 tuple 返回
        return {k: tuple(v) for k, v in pre_tokens.items()}

    # ------------------------------------------------------------------
    # 重写 encode: 先收集 unique pretoken, 再批量 merge, 最后按序列展开
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_pattern:
            chunks = re.split(self.special_pattern, text)
        else:
            chunks = [text]

        # ① 收集本块所有 unique pretoken, 并记录出现顺序
        pre_tokens = {}          # {pretoken_bytes: [单字节, ...]}
        pre_tokens_seq = []      # 按出现顺序: bytes(普通) 或 str(特殊)

        for chunk in chunks:
            if not chunk:
                continue
            if self.special_tokens and chunk in self.special_tokens:
                pre_tokens_seq.append(chunk)      # str 标记特殊 token
                continue
            for match in re.finditer(PAT, chunk):
                b_word = match.group().encode('utf-8')
                if b_word not in pre_tokens:
                    pre_tokens[b_word] = [b_word[i:i+1] for i in range(len(b_word))]
                pre_tokens_seq.append(b_word)      # bytes 普通 pretoken

        # ② 批量 merge (unique pretoken 只算一次)
        merged_map = self.merge_pretokens_batch(pre_tokens)

        # ③ 按序列展开成 id
        for item in pre_tokens_seq:
            if isinstance(item, str):              # 特殊 token
                ids.append(self.vocab_to_id[item.encode('utf-8')])
            else:                                  # 普通 pretoken
                for token in merged_map[item]:
                    ids.append(self.vocab_to_id[token])

        return ids

    # ------------------------------------------------------------------
    # encode_iterable 保持原样（偏移已修好）
    # ------------------------------------------------------------------
    def encode_iterable(self, iterable):
        """
        流式编码：跨 chunk 复用 cache，重复 pretoken 全局只 merge 一次。
        """
        chunk_size = 1 * 1024 ** 2
        cache = {}                       # {pretoken_bytes: tuple}
        file_pos = 0

        max_special_len = len(max(self.special_tokens, key=len)) if self.special_tokens else 0

        while True:
            chunk = iterable.read(chunk_size)
            if not chunk:
                break

            is_last_chunk = len(chunk) < chunk_size
            original_chunk = chunk

            # ---- 截断：保证不切断特殊 token / 最后一个 pretoken ----
            if not is_last_chunk:
                if self.special_tokens:
                    concat_special_tokens = "|".join(re.escape(t) for t in self.special_tokens)
                    end_chunk = chunk[-2 * max_special_len:]
                    matches = list(re.finditer(concat_special_tokens, end_chunk))
                    if matches:
                        offset = len(chunk) - len(end_chunk)
                        chunk = chunk[:offset + matches[0].span()[0]]
                    else:
                        chunk = chunk[:-max_special_len]
                        pretokens = re.findall(PAT, chunk)
                        if pretokens:
                            chunk = chunk[:-len(pretokens[-1])]
                else:
                    pretokens = re.findall(PAT, chunk)
                    if pretokens:
                        chunk = chunk[:-len(pretokens[-1])]

                if not chunk:
                    chunk = original_chunk       # 兜底：截没了就退回原块

            chunk_len = len(chunk)
            if chunk_len == 0:
                break

            # 指针推进：直接从 file_pos 推 chunk_len
            file_pos += len(chunk.encode(iterable.encoding or "utf-8"))
            iterable.seek(file_pos)

            for tid in self._encode_with_cache(chunk, cache):
                yield tid
    # ------------------------------------------------------------------
    # decode 保持原样
    # ------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        for i in ids:
            if i in self.vocab:
                byte_parts.append(self.vocab[i])
            else:
                byte_parts.append(b'')

        b = b"".join(byte_parts)
        return b.decode("utf-8", errors="replace")