import numpy as np
from collections import Counter

import matplotlib.pyplot as plt

import random
from glob import glob
from tqdm.auto import tqdm
import math


def load_data():
    with open("data/eng-input.txt", "r") as f:
        data = f.read().replace("\n", "")

    return data


def cut_words(data, s):
    s_idx = np.where(s)[0] + 1
    N = len(data)

    idx_start = np.hstack([np.array([0]), s_idx])
    idx_end = np.hstack([s_idx, np.array([N])])

    words = [data[start:end] for start, end in zip(idx_start, idx_end)]

    return words


class Model:
    def __init__(self, alpha = 100, p_c = 0.5):
        self.alpha = alpha
        self.p_c = p_c

    def P0(self, w_len):
        U = 1.0/float(self.C)

        return U**w_len * self.p_c**(w_len-1)

    def PW(self, w, w_len, a=0, b=0):
        return (self.alpha * self.P0(w_len) + self.count[w] + a) / (self.alpha + self.word_count + b)

    def log_P_data(self, data, s):
        return sum([np.log(self.PW(w, len(w))) for w in cut_words(data, s)])

    def fit(self, data, num_iter=1000):
        self.data = data
        self.N = len(data)
        self.s = np.random.randint(0, 2, size=self.N - 2)

        words = cut_words(data, self.s)

        self.count = Counter(words)
        self.word_count = sum(self.count.values())

        self.C = len(set(data))

        p_data = []
        ss = []

        inner_i = 0

        total_iters = num_iter * (self.N - 2)
        print_every = total_iters // 500

        for iter in tqdm(range(num_iter)):
            for i in np.random.permutation(self.N - 2):
                inner_i += 1

                prev_left = 0
                for j in reversed(range(i)):
                    if self.s[j] == 1:
                        prev_left = j
                        break

                next_right = len(self.s)
                for j in range(i + 1, len(self.s)):
                    if self.s[j] == 1:
                        next_right = j
                        break

                prev_word = data[prev_left:i]
                prev_len = i - prev_left

                next_word = data[i:next_right]
                next_len = next_right - i

                joined_word = prev_word + next_word
                joined_len = next_right - prev_left

                if self.s[i] == 0:
                    self.count[joined_word] = max(0, self.count[joined_word] - 1)
                    self.word_count -= 1
                else:
                    self.count[prev_word] = max(0, self.count[prev_word] - 1)
                    self.count[next_word] = max(0, self.count[next_word] - 1)
                    self.word_count -= 2

        #         p0 = (alpha * P0(joined_word) + count[joined_word]) / (alpha + word_count)

        #         a = (alpha * P0(prev_word) + count[prev_word]) / (alpha + word_count)
        #         b = (alpha * P0(next_word) + count[next_word]) / (alpha + word_count + 1)
        #         p1 =  a * b

                p0 = self.PW(joined_word, joined_len)
                p1 = self.PW(prev_word, prev_len) * self.PW(next_word, next_len, b=1)

        #         assert np.all(np.fromiter(count.values(), dtype=np.int32) >= 0)

                if (random.random() * (p0 + p1)) < p1:
                    self.s[i] = 0
                else:
                    self.s[i] = 1

                if self.s[i] == 0:
                    self.count[joined_word] += 1
                    self.word_count += 1
                else:
                    self.count[prev_word] += 1
                    self.count[next_word] += 1
                    self.word_count += 2

                if inner_i % print_every == 0:
                    p_data.append(self.log_P_data(data, self.s))
                    ss.append(self.s.copy())

        self.final_log_P_data = self.log_P_data(data, self.s)
        self.history_p_data = p_data
        self.history_s = ss

        return self


    def plot_results(self):
        print(self.final_log_P_data)

        plt.figure(figsize=(15, 4))
        plt.subplot(131)
        plt.imshow(np.array(self.history_s), aspect="auto")

        plt.subplot(132)
        plt.plot(np.sum(self.history_s - np.roll(self.history_s, 1), axis=1))

        plt.subplot(133)
        plt.plot(self.history_log_P_data)

