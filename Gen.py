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

    def get_some_logits(self, u, sample=[]):
        logits = np.array([0.] * self.itemNum)
        logits[sample] = np.sum(np.multiply(self.users_embedding[u], self.items_embedding[sample]), 1) + \
                         self.items_bias[sample]
        return logits

    def updates(self, u, lambda_p, lambda_1, lambda_2, lambda_3, pos, sample):
        ratings=[]
        ratings=self.get_all_logits(u=u,all_logits=ratings)
        exp_ratings=np.exp(ratings)#exp ratings
        sum_exprating=np.sum(exp_ratings)#sum of exp ratings
        sum_expr_square=pow(sum_exprating,2)

        X = np.array([0.] * self.itemNum)
        X[sample] = exp_ratings[sample] / sum_exprating

        ##to calculate the partial_L_X
        partial_L_X = np.array([0.]*self.itemNum)
        lambda_3[pos] =lambda_1[pos]*X[pos]/(lambda_2*X[pos]+(1-lambda_2)*(1/len(pos)))
        partial_L_X[sample]=lambda_3[sample]/X[sample]

        #print(partial_L_X[sample])
        ##delta for U_u
        delta_for_u = np.array([0.]*self.emb_dim)
        delta_u=np.array([0.]*self.emb_dim)
        tmp2 = np.array([0.] * self.emb_dim)  # initalize as 0

        for j in range(self.itemNum):
            tmp2 += exp_ratings[j] * np.array(self.items_embedding[j])

        for i in sample:
            tmp=exp_ratings[i]*np.array(self.items_embedding[i])*sum_exprating - exp_ratings[i]*tmp2
            partial_X_U=tmp/sum_expr_square
            delta_for_u+=partial_L_X[i] * partial_X_U

        delta_u=(-1/len(sample))* np.array(delta_for_u) + self.alpha_u*np.array(self.users_embedding[u])
        #print('delta_u:',delta_u)

        ##there are some common part between partial_x to partial_V_k and partial_x to partial_b_k
        delta_L_b=np.array([0.]*self.itemNum)
        delta_L_V=np.array([([0.]*self.emb_dim) for i in range(self.itemNum)])
        partial_X_b = np.array([0.]*self.itemNum)
        
        sample_list=list(sample)
        for s in sample:
            for i in range(self.itemNum):
                partial_X_b[i] = (-exp_ratings[i] * exp_ratings[s]) / sum_expr_square
                if s == i:
                    partial_X_b[i] += (exp_ratings[i] * sum_exprating) / sum_expr_square
                delta_L_b[i] += partial_L_X[s] * partial_X_b[i]
                delta_L_V[i] += delta_L_b[i] * self.users_embedding[u]

        delta_L_V = (-1/len(sample)) * np.array(delta_L_V) + self.alpha_v * np.array(self.items_embedding)
        delta_L_b = (-1/len(sample)) * np.array(delta_L_b) + self.beta_v * np.array(self.items_bias)

        ##update simultaneously for the user and sample items
        self.users_embedding[u] -= self.learning_rate * delta_u
        self.items_embedding -= self.learning_rate * delta_L_V
        self.items_bias -= self.learning_rate * delta_L_b

    def get_user_all_rating(self, user):
        logit = np.sum(np.multiply(self.users_embedding[user], self.items_embedding), 1) + self.items_bias
        return logit
        # return logit

    def save_model(self,best_g_users_embedding,best_g_items_embedding,best_g_items_bias):
        with open('./ml-100k/gen_model', 'a')as fout:
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
