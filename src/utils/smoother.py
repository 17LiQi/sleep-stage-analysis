# src/utils/smoother.py
import numpy as np


class ViterbiSmoother:
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.trans_mat = None
        self.initial_dist = None
        self.log_trans_mat = None
        self.log_initial_dist = None


    def fit(self, labels_sequence_list):
        """从真实的标签序列中学习转移概率和初始分布"""
        trans_counts = np.zeros((self.n_classes, self.n_classes))
        initial_counts = np.zeros(self.n_classes)

        valid_sequences = [seq for seq in labels_sequence_list if seq]
        if not valid_sequences:
            print("警告: Viterbi fit 方法收到了空的标签序列列表，无法训练。")
            return

        for seq in valid_sequences:
            initial_counts[seq[0]] += 1
            for i in range(len(seq) - 1):
                if seq[i] < self.n_classes and seq[i+1] < self.n_classes:
                    trans_counts[seq[i], seq[i + 1]] += 1

        smoothing_factor = 1.0 # 拉普拉斯平滑

        if initial_counts.sum() == 0:
            self.initial_dist = np.ones(self.n_classes) / self.n_classes
        else:
            self.initial_dist = (initial_counts + smoothing_factor) / (initial_counts.sum() + self.n_classes * smoothing_factor)

        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.trans_mat = (trans_counts + smoothing_factor) / (row_sums + self.n_classes * smoothing_factor)

        # 计算并保存对数概率
        self.log_trans_mat = np.log(self.trans_mat + 1e-9)
        self.log_initial_dist = np.log(self.initial_dist + 1e-9)

        print("维特比平滑器已训练完成。")
        # print("转移概率矩阵:\n", self.trans_mat) # 可以选择性打印

    def predict(self, log_probabilities_sequence):
        """使用维特比算法解码最可能的标签路径。"""
        if self.log_initial_dist is None or self.log_trans_mat is None:
            raise RuntimeError("ViterbiSmoother 必须在调用 predict 之前先调用 fit 方法进行训练。")

        n_epochs = len(log_probabilities_sequence)
        if n_epochs == 0:
            return np.array([], dtype=int)

        T1 = np.zeros((n_epochs, self.n_classes))
        T2 = np.zeros((n_epochs, self.n_classes), dtype=int)

        T1[0, :] = self.log_initial_dist + log_probabilities_sequence[0, :]

        for t in range(1, n_epochs):
            for j in range(self.n_classes):
                probs = T1[t - 1, :] + self.log_trans_mat[:, j] + log_probabilities_sequence[t, j]
                T2[t, j] = np.argmax(probs)
                T1[t, j] = np.max(probs)

        z = np.zeros(n_epochs, dtype=int)
        z[n_epochs - 1] = np.argmax(T1[n_epochs - 1, :])
        for t in range(n_epochs - 2, -1, -1):
            z[t] = T2[t + 1, z[t + 1]]

        return z