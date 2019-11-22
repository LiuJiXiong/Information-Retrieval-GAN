import random

import numpy as np


class Generative():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param, initdelta, learning_rate):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.alpha_u = self.alpha_v = self.beta_v = lamda
        self.users_embedding = np.array([([0.] * self.emb_dim) for i in range(self.userNum)])
        self.items_embedding = np.array([([0.] * self.emb_dim) for i in range(self.itemNum)])
        self.items_bias = np.array([0.] * self.itemNum)

        ## init the user latent feature matrix and item latent feature matrix
        if (param == None):
            i = 0
            while (i < self.userNum):
                j = 0
                while (j < self.emb_dim):
                    self.users_embedding[i][j] = random.uniform(-self.initdelta, self.initdelta)
                    j += 1
                i += 1

            i = 0
            while (i < self.itemNum):
                j = 0
                while (j < self.emb_dim):
                    self.items_embedding[i][j] = random.uniform(-self.initdelta, self.initdelta)
                    j += 1
                i += 1
            self.items_bias = np.array([0.] * self.itemNum)
        else:
            self.users_embedding = self.param[0]
            self.users_embedding = np.array(self.users_embedding)
            self.items_embedding = self.param[1]
            self.items_embedding = np.array(self.items_embedding)
            self.items_bias = self.param[2]
            self.items_bias = np.array(self.items_bias)

    def get_all_logits(self, u, all_logits):
        all_items = np.arange(self.itemNum)
        all_logits = np.sum(np.multiply(self.users_embedding[u], self.items_embedding[all_items]), 1) + self.items_bias[
            all_items]
        return all_logits

    def get_user_all_rating(self, user):
        logit = np.sum(np.multiply(self.users_embedding[user], self.items_embedding), 1) + self.items_bias
        return logit
        # return logit


    def updates(self, u, i_pos, i_neg):
        pos_rating=np.dot(self.users_embedding[u],self.items_embedding[i_pos])+self.items_bias[i_pos]
        neg_rating=np.dot(self.users_embedding[u],self.items_embedding[i_neg])+self.items_bias[i_neg]
        delta=1/(1+np.exp(neg_rating-pos_rating)) -1
        partial_L_U=delta*(self.items_embedding[i_pos]-self.items_embedding[i_neg])+self.alpha_u*self.users_embedding[u]
        
        partial_L_Vi=delta*self.users_embedding[u]+self.alpha_v*self.items_embedding[i_pos]
        partial_L_bi=delta+self.beta_v*self.items_bias[i_pos]

        partial_L_Vj=-delta*self.users_embedding[u]+self.alpha_v*self.items_embedding[i_neg]
        partial_L_bj=-delta+self.beta_v*self.items_bias[i_neg]

        ##update simultaneously for the user and positive item and negative item
        self.users_embedding[u] -= self.learning_rate * partial_L_U
        self.items_embedding[i_pos] -= self.learning_rate * partial_L_Vi
        self.items_bias[i_pos] -= self.learning_rate * partial_L_bi

        self.items_embedding[i_neg] -= self.learning_rate*partial_L_Vj
        self.items_bias[i_neg] -= self.learning_rate*partial_L_bj

    def save_model(self,best_g_users_embedding,best_g_items_embedding,best_g_items_bias):
        with open('./ml-100k/gen_model_init', 'a')as fout:
            for u in range(self.userNum):
                fout.write(str(best_g_users_embedding[u]) + '\n')
            for i in range(self.itemNum):
                fout.write(str(best_g_items_embedding[i]) + '\n')
            for i in range(self.itemNum):
                fout.write(str(best_g_items_bias[i]) + ' ')
            fout.write('\n')
            fout.write('\n')
            fout.flush()
            fout.close()
