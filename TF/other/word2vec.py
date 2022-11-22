"""
训练词向量模型
"""
import collections
import random
import zipfile

import numpy as np
import tensorflow as tf

# 训练参数
# 学习率
learning_rate = 0.1
batch_size = 128
# 学习次数
num_steps = 3000000
# 多少次做一下显示
display_step = 10000
# 多少次做一下验证
eval_step = 200000

# 测试样例
eval_words = ['nine', 'of', 'going', 'hardware', 'american', 'britain']

# Word2Vec 参数
# 词向量维度
embedding_size = 200
# 语料库词语数
max_vocabulary_size = 50000
# 最小词频
min_occurrence = 10
# 左右窗口大小,即一次取7个词
skip_window = 3
# 一次制作多少个输入输出对
num_skips = 2
# 负采样
num_sampled = 64

# 加载训练数据
data_path = './data/text8.zip'
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()
print(len(text_words))

# 创建一个计数器，计算每个词出现了多少次
# 将非常用词设置成UNK,并初始化成-1个
count = [('UNK', -1)]
# 基于词频返回max_vocabulary_size个常用词
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
# Counter的结果默认是从大到小排序的,打印出现次数前十的
print(count[0:10])

# 剔除掉出现次数少于'min_occurrence'的词
# 从后往前一个个查(从小到大)
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        # 判断时，从小到大排序的，所以跳出时候剩下的都是满足条件的
        break

# 将每个词映射到一个id
# 计算语料库大小,因为根据min_occurrence重新筛过一遍,总数量是小于max_vocabulary_size的
vocabulary_size = len(count)
# 每个词都分配一个ID
word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i

# print(word2id)

# 将文本里的词全部转换成id的形式
data = list()
unk_count = 0
for word in text_words:
    # 全部转换成id
    id_value = word2id.get(word, 0)
    if id_value == 0:
        # 非频繁词
        unk_count += 1
    data.append(id_value)
count[0] = ('UNK', unk_count)
# id到字符的映射,将字符到id的映射做个翻转
id2word = dict(zip(word2id.values(), word2id.keys()))

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])

data_index = 0


# 获取训练数据的函数,一个批次一个批次的生成数据
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 7为整个窗口大小，左3右3中间1
    span = 2 * skip_window + 1
    # 创建一个长度为7的队列
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):  # 如果数据被滑完一遍了
        data_index = 0
    buffer.extend(data[data_index:data_index + span])  # 队列里存的是当前窗口，例如deque([5234, 3081, 12, 6, 195, 2, 3134], maxlen=7)
    data_index += span
    for i in range(batch_size // num_skips):  # num_skips表示取多少组不同的词作为输出，此例为2
        context_words = [w for w in range(span) if w != skip_window]  # 上下文就是[0, 1, 2, 4, 5, 6]
        words_to_use = random.sample(context_words, num_skips)  # 在上下文里随机选2个候选词
        for j, context_word in enumerate(words_to_use):  # 遍历每一个候选词，用其当做输出也就是标签
            batch[i * num_skips + j] = buffer[skip_window]  # 输入都为当前窗口的中间词，即3
            labels[i * num_skips + j, 0] = buffer[context_word]  # 用当前候选词当做标签
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            # 之前已经传入7个词了，窗口要右移了，例如原来为[5234, 3081, 12, 6, 195, 2, 3134]，现在为[3081, 12, 6, 195, 2, 3134, 46]
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


with tf.device('/cpu:0'):
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))  # 维度：47135, 200
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


def get_embedding(x):
    with tf.device('/cpu:0'):
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed


def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=y,
                           inputs=x_embed,
                           num_sampled=num_sampled,  # 采样出多少个负样本
                           num_classes=vocabulary_size))
        return loss


# Evaluation.
def evaluate(x_embed):
    with tf.device('/cpu:0'):
        # Compute the cosine similarity between input data embedding and every embedding vectors
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))  # 归一化
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32)  # 全部向量的
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)  # 计算余弦相似度
        return cosine_sim_op


# SGD
optimizer = tf.optimizers.SGD(learning_rate)


# 迭代优化
def run_optimization(x, y):
    with tf.device('/cpu:0'):
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = nce_loss(emb, y)

        # 计算梯度
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])

        # 更新
        optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))


# 待测试的几个词
x_test = np.array([word2id[w.encode('utf-8')] for w in eval_words])

# 训练
for step in range(1, num_steps + 1):
    batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
    run_optimization(batch_x, batch_y)

    if step % display_step == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))

    # Evaluation.
    if step % eval_step == 0 or step == 1:
        print("Evaluation...")
        sim = evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k = 8  # 返回前8个最相似的
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % eval_words[i]
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, id2word[nearest[k]])
            print(log_str)
