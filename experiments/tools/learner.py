import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
"""
    sup should have two methods: intended_action() and sample_action()
    which return the intended action and the potentially noisy action respectively.
"""

class Learner():

    def __init__(self, est, bootstrap_ratio=1, sup=None):
        self.X = []
        self.y = []
        self.est = est
        self.N = len(self.est)
        self.bootstrap_ratio = bootstrap_ratio

    def add_data(self, states, actions):
        assert type(states) == list
        assert type(actions) == list
        self.X += states
        self.y += actions

    def clear_data(self):
        self.X = []
        self.y = []

    def train(self, verbose=False):
        X_train, y_train = np.array(self.X), np.array(self.y)

        idxs = np.arange(len(X_train))
        bootstrapped_idxs = [np.random.choice(idxs, size=int(self.bootstrap_ratio*len(X_train)), replace=True) for _ in range(self.N)]
        X_train_bootstrapped = [X_train[bootstrapped_idxs[i]] for i in range(self.N)]
        y_train_bootstrapped = [y_train[bootstrapped_idxs[i]] for i in range(self.N)]
        histories = [self.est[i].fit(X_train_bootstrapped[i], y_train_bootstrapped[i]) for i in range(self.N)]

        if verbose == True:
            scores = [self.est[i].score(X_train_bootstrapped[i], y_train_bootstrapped[i]) for i in range(self.N)]
            mean_score = np.mean(scores, axis=0)
            var_score = np.var(scores, axis=0)
            print "Train score mean: " + str(mean_score), " Train score var: " + str(var_score)
        return histories

    def acc(self):
        return np.mean([self.est[i].score(self.X, self.y) for i in range(self.N)], axis=0)

    def intended_actions(self, s):
        return [self.est[i].predict([s])[0] for i in range(self.N)]

    def intended_action(self, s):
        return np.mean(self.intended_actions(s), axis=0)

    def sample_action(self, s):
        return self.intended_action(s)



