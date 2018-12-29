import numpy as np
from collections import Counter
from unidecode import unidecode

import matplotlib.pyplot as plt

import random
from glob import glob
from tqdm.auto import tqdm
import math


def load_data(strip_unicode=False):
    with open("data/eng-input.txt", "r") as f:
        data = f.read().replace("\n", "")

    if strip_unicode:
        return unidecode(data)
    else:
        return data


def cut_words(data, s):
    s_idx = np.where(s)[0]
    N = len(data)

    idx_start = np.hstack([np.array([0]), s_idx])
    idx_end = np.hstack([s_idx, np.array([N])])

    words = [data[start:end] for start, end in zip(idx_start, idx_end)]

    return words


class Model:
    def __init__(self, alpha = 100, p_c = 0.5):
        self.alpha = alpha
        self.p_c = p_c

    def fit(self, data, num_iter=1000):
        self.data = data
        N = len(data)
        s = np.random.randint(0, 2, size=N)
        alpha = self.alpha
        p_c = self.p_c

        words = cut_words(data, s)

        count = Counter(words)
        word_count = sum(count.values())

        C = len(set(data))

        def P0(w):
            U = 1.0/float(C)

            return U**len(w) * self.p_c**(len(w)-1)

        def PW(w, a=0, b=0):
            return (alpha * P0(w) + count[w] + a) / (alpha + word_count + b)

        def log_P_data():
            return sum([np.log(PW(w)) for w in cut_words(data, s)])

        def data_entropy():
            ps = [PW(w) for w in cut_words(data, s)]
            return -np.sum(ps * np.log(ps))

        self.history_p_data = []
        self.history_entropy = []
        self.history_s = []

        inner_i = 0

        total_iters = num_iter * (N)
        print_every = total_iters // 100

        for iter in tqdm(range(num_iter)):
            for i in np.random.permutation(N):
                inner_i += 1

                if i == 0 or i == N - 1:
                    continue

                prev_left = 0
                for j in reversed(range(i)):
                    if s[j] == 1:
                        prev_left = j
                        break

                next_right = len(s)
                for j in range(i + 1, len(s)):
                    if s[j] == 1:
                        next_right = j
                        break

                prev_word = data[prev_left:i]
                next_word = data[i:next_right]
                joined_word = prev_word + next_word

                if s[i] == 0:
                    assert count[joined_word] > 0, f"got count 0 for {joined_word} = {prev_word} + {next_word}"

                    count[joined_word] -= 1
                    word_count -= 1
                else:
                    assert count[prev_word] > 0 and count[next_word] > 0, \
                            f"got count 0 for {joined_word} = {prev_word} + {next_word}"

                    count[prev_word] -= 1
                    count[next_word] -= 1
                    word_count -= 2

                p0 = PW(joined_word)
                p1 = PW(prev_word) * PW(next_word, b=1)

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

