import torch
from .strategy import Strategy
from sklearn.neural_network import MLPClassifier

class DiscriminativeSampling(Strategy):
    def __init__(self, dataset, net):
        super(DiscriminativeSampling, self).__init__(dataset, net)
        self.clf = MLPClassifier(hidden_layer_sizes=(256, ), early_stopping=True)

    def query(self, n):
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()
        self.clf.fit(embeddings, labeled_idxs)
        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        unlabeled_probs = self.clf.predict_proba(embeddings[~labeled_idxs])[:, 0]
        return unlabeled_idxs[(-unlabeled_probs).argsort()[:n]]
