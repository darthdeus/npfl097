# cython: language_level=3
import numpy as np
from collections import Counter, defaultdict
from unidecode import unidecode

import matplotlib.pyplot as plt

import random
from glob import glob
from tqdm.auto import tqdm
import math


def load_data():
    with open("data/eng-input.txt", "r") as f:
        data = f.read().replace("\n", "")

        # TODO: remove this later!
        return unidecode(data)


def cut_words(data, s):
    s_idx = np.where(s)[0]
    N = len(data)

    idx_start = np.hstack([np.array([0]), s_idx])
    idx_end = np.hstack([s_idx, np.array([N])])

    words = [data[start:end] for start, end in zip(idx_start, idx_end)]

    return words

cdef P0(str w, float C, float p_c):
    cdef float U = 1.0 / C
    cdef int l = len(w)

    return U**l * p_c**(l-1)

cdef PW(str w, float alpha, int word_count, float C, float p_c, int count_w, int a=0, int b=0):
    cdef float p = P0(w, C, p_c)
    cdef float num = (alpha * p + count_w + a)
    cdef float den = (alpha + word_count + b)
    return num / den

class Model:
    def __init__(self, alpha = 100, p_c = 0.5):
        self.alpha = alpha
        self.p_c = p_c


    def fit(self, data, num_iter=1000):
        self.data = data
        cdef int N = len(data)
        s = np.random.randint(0, 2, size=N)
        cdef float alpha = self.alpha
        cdef float p_c = self.p_c

        words = cut_words(data, s)

        count = Counter(words)
        cdef int word_count = sum(count.values())

        cdef float C = len(set(data))

        def log_P_data():
            return sum([np.log(PW_slow(w)) for w in cut_words(data, s)])

        def data_entropy():
            ps = [PW_slow(w) for w in cut_words(data, s)]
            return -np.sum(ps * np.log(ps))

        def PW_slow(w, a = 0, b = 0):
            return (alpha * P0(w, C, p_c) + count[w] + a) / (alpha + word_count + b)

        self.history_p_data = []
        self.history_entropy = []
        self.history_s = []

        total_iters = num_iter * (N)
        print_every = total_iters // 100

        cdef int s_len = len(s)

        def loop(word_count, s):
            cdef int inner_i = 0
            cdef int prev_left = 0
            cdef int next_right = s_len

            for _ in tqdm(range(num_iter)):
                for i in np.random.permutation(N):
                    inner_i += 1

                    if i == 0 or i == N - 1:
                        continue

                    for x in range(10):
                        s[i] += x

                    prev_left = 0
                    for j in reversed(range(i)):
                        if s[j] == 1:
                            prev_left = j
                            break

                    next_right = s_len
                    for j in range(i + 1, s_len):
                        if s[j] == 1:
                            next_right = j
                            break

                    prev_word = data[prev_left:i]
                    next_word = data[i:next_right]
                    joined_word = prev_word + next_word

                    if s[i] == 0:
                        # assert count[joined_word] > 0, f"got count 0 for {joined_word} = {prev_word} + {next_word}"

                        count[joined_word] -= 1
                        # count[joined_word] = max(0, count[joined_word] - 1)
                        word_count -= 1
                    else:
                        # assert count[prev_word] > 0 and count[next_word] > 0, \
                        #         f"got count 0 for {joined_word} = {prev_word} + {next_word}"

                        count[prev_word] -= 1
                        count[next_word] -= 1
                        word_count -= 2

            #         p0 = (alpha * P0(joined_word) + count[joined_word]) / (alpha + word_count)
            #         a = (alpha * P0(prev_word) + count[prev_word]) / (alpha + word_count)
            #         b = (alpha * P0(next_word) + count[next_word]) / (alpha + word_count + 1)
            #         p1 =  a * b

                    p0 = PW(joined_word, alpha, word_count, C, p_c, count[joined_word])
                    p1 = PW(prev_word, alpha, word_count, C, p_c, count[prev_word]) * PW(next_word, alpha, word_count, C, p_c, count[next_word], a=0, b=1)

                    # assert np.all(np.fromiter(count.values(), dtype=np.int32) >= 0)

                    if (random.random() * (p0 + p1)) < p1:
                        s[i] = 0
                    else:
                        s[i] = 1

                    if s[i] == 0:
                        count[joined_word] += 1
                        word_count += 1
                    else:
                        count[prev_word] += 1
                        count[next_word] += 1
                        word_count += 2

                    if inner_i % print_every == 0:
                        self.history_p_data.append(log_P_data())
                        self.history_entropy.append(data_entropy())
                        self.history_s.append(s.copy())

        loop(word_count, s)

        self.s = s
        self.entropy = data_entropy()
        self.final_log_P_data = log_P_data()

        return self


    def plot_results(self):
        print(self.final_log_P_data)

        plt.figure(figsize=(15, 4))
        plt.subplot(131)
        plt.imshow(np.array(self.history_s), aspect="auto")

        plt.subplot(132)
        plt.plot(self.history_entropy)
        plt.title(f"Final entropy {self.history_entropy[-1]}")

        plt.subplot(133)
        plt.plot(self.history_p_data[2:])
        plt.title(f"Final log_p {self.history_p_data[-1]}")

        plt.show()

        return " ".join(cut_words(self.data, self.s))[:2000]

