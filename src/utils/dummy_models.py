"""占位模型类，用于生成可反序列化的 dummy 模型文件供测试使用。"""

class DummyModel:
    def __init__(self, n_features=3):
        self.n_features_in_ = n_features

    def predict(self, X):
        return [1 if sum(x) > 0 else 0 for x in X]

    def predict_proba(self, X):
        probs = []
        for x in X:
            s = float(sum(x))
            # 简单 sigmoid
            import math

            p = 1.0 / (1.0 + math.exp(-s))
            probs.append([1 - p, p])
        return probs


class DummyScaler:
    def transform(self, X):
        return X
