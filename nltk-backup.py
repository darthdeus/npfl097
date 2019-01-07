def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)

def perturb(text, p):
    alphabet = list(set(text_en) - set(["\n"]))

    return "".join([c
                    if (random.random() > p or c == "\n")
                    else random.choice(alphabet)
                    for c in text])

unigram = []
bigram = []
cond_bigram = []

for p in X:
    e1 = []
    e2 = []
    e3 = []

    for i in range(20):
        text = perturb(text_en, p).split("\n")

        e1.append(entropy(text))
        e2.append(entropy(nltk.bigrams(text)))
        e3.append(entropy(nltk.bigrams(text)) - entropy(text))

    unigram.append(np.mean(e1))
    bigram.append(np.mean(e2))
    cond_bigram.append(np.mean(e3))

plt.plot(X, unigram, label="unigram")
plt.plot(X, bigram, label="bigram")
plt.plot(X, cond_bigram, label="cond-bigram")
plt.legend()



def perturb(text, p):
    alphabet = list(set(text_en)) # - set(["\n"]))

    return "".join([c
                    if (random.random() > p)# or c == "\n")
                    else random.choice(alphabet)
                    for c in text])

unigram = []
bigram = []
cond_bigram = []

for p in X:
    e1 = []
    e2 = []
    e3 = []

    for i in range(20):
        text = perturb(text_en, p).split("\n")

        e1.append(entropy(text))
        e2.append(entropy(nltk.bigrams(text)))
        e3.append(entropy(nltk.bigrams(text)) - entropy(text))

    unigram.append(np.mean(e1))
    bigram.append(np.mean(e2))
    cond_bigram.append(np.mean(e3))

plt.plot(X, unigram, label="unigram")
plt.plot(X, bigram, label="bigram")
plt.plot(X, cond_bigram, label="cond-bigram")
plt.legend()



def ngram_list(text,n):
    ngram=[]
    count=0
    for token in text[:len(text)-n+1]:
        ngram.append(text[count:count+n])
        count=count+1
    return ngram

def condentropy(data):
    def ngram_counts(text, n):
        ngram_d = {}
        ngram_l = ngram_list(text, n)

        for item in ngram_l:
            ngram_d[' '.join(item)] = (ngram_d[' '.join(item)] + 1) if ' '.join(item) in ngram_d else 1
        return ngram_d

    uni_gram = ngram_counts(data, 1)
    bi_gram = ngram_counts(data, 2)
    N = sum(uni_gram.values())
    H = 0

    for key in bi_gram.keys():
        H -= bi_gram[key] / (1.0 * N) * math.log(bi_gram[key] / (1.0 * uni_gram[key.split(' ')[1]]),2)

    return H
