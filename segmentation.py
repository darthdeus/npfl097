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


def fit(data, alpha = 100, p_c = 0.5, num_iter = 1000):
    N = len(data)
    s = np.random.randint(0, 2, size=N - 2)

    words = cut_words(data, s)

    count = Counter(words)
    word_count = sum(count.values())

    C = len(set(data))

    def P0(w):
        U = 1.0/float(C)

        return U**len(w) * p_c**(len(w)-1)

    def PW(w, a=0, b=0):
        return (alpha * P0(w) + count[w] + a) / (alpha + word_count + b)

    def log_P_data(data, s):
        return sum([np.log(PW(w)) for w in cut_words(data, s)])

    p_data = []
    ss = []

    inner_i = 0

    total_iters = num_iter * (N-2)
    print_every = total_iters // 500

    for iter in tqdm(range(num_iter)):
        for i in np.random.permutation(N - 2):
            inner_i += 1

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
                count[joined_word] = max(0, count[joined_word] - 1)
                word_count -= 1
            else:
                count[prev_word] = max(0, count[prev_word] - 1)
                count[next_word] = max(0, count[next_word] - 1)
                word_count -= 2

    #         p0 = (alpha * P0(joined_word) + count[joined_word]) / (alpha + word_count)

    #         a = (alpha * P0(prev_word) + count[prev_word]) / (alpha + word_count)
    #         b = (alpha * P0(next_word) + count[next_word]) / (alpha + word_count + 1)
    #         p1 =  a * b

            p0 = PW(joined_word)
            p1 = PW(prev_word) * PW(next_word, b=1)

    #         assert np.all(np.fromiter(count.values(), dtype=np.int32) >= 0)

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
                p_data.append(log_P_data(data, s))
                ss.append(s.copy())

    return log_P_data(data, s), s, p_data, ss


def plot_results(log_P, s, p_data, ss):
    print(log_P)

    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(np.array(ss), aspect="auto")

    plt.subplot(132)
    plt.plot(np.sum(ss - np.roll(ss, 1), axis=1))

    plt.subplot(133)
    plt.plot(p_data[10:])

