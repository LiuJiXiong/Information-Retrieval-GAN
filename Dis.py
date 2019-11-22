import random

import numpy as np


class Discriminative():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param, initdelta, learning_rate):
        self.itemNum=itemNum
        self.userNum=userNum
        self.emb_dim=emb_dim
        self.lamda=lamda
        self.param=param
        self.initdelta=initdelta
        self.learning_rate=learning_rate
        self.alpha_U=self.alpha_V=self.beta_V = lamda
        self.users_embedding = np.array([([0.] * self.emb_dim) for i in range(self.userNum)])
        self.items_embedding = np.array([([0.] * self.emb_dim) for i in range(self.itemNum)])
        self.items_bias = np.array([0.]*self.itemNum)
        ## init the user latent feature matrix and item latent feature matrix

        if(param==None):
            i=0
            while(i<self.userNum):
                j=0
                while(j<self.emb_dim):
                    self.users_embedding[i][j]=random.uniform(-self.initdelta,self.initdelta)
                    j+=1
                i+=1

            i = 0
            while (i < self.itemNum):
                j = 0
                while (j < self.emb_dim):
                    self.items_embedding[i][j] = random.uniform(-self.initdelta, self.initdelta)
                    j += 1
                i += 1
            self.items_bias=np.array([0.]*self.itemNum)
        else:
            self.users_embedding=self.param[0]
            self.users_embedding=np.array(self.users_embedding)
            self.items_embedding=self.param[1]
            self.items_embedding=np.array(self.items_embedding)
            self.items_bias=self.param[2]
            self.items_bias = np.array(self.items_bias)

    
    def updates(self,u_sample,i_sample,label_sample):
        rating=np.sum(np.multiply(self.users_embedding[u_sample],self.items_embedding[i_sample]),1)+self.items_bias[i_sample]
        rating=np.array(rating)
        f=np.array([0.]*len(u_sample))
        for i in range(len(rating)):
            if(rating[i]>0):
                f[i] = 1 - label_sample[i] - np.exp(-rating[i])/(1+np.exp(-rating[i]))
            else:
                f[i] = -label_sample[i] + np.exp(rating[i])/(1+np.exp(rating[i]))

        delta_U = np.array([([0.] * self.emb_dim) for i in range(self.userNum)])
        delta_V = np.array([([0.] * self.emb_dim) for i in range(self.itemNum)])
        delta_Bi= np.array([0.]*self.itemNum)

        for i in range(len(u_sample)):
            delta_U[u_sample[i]] += f[i]*self.items_embedding[i_sample[i]] + self.alpha_U * self.users_embedding[u_sample[i]]
            delta_V[i_sample[i]] += f[i]*self.users_embedding[u_sample[i]] + self.alpha_V * self.items_embedding[i_sample[i]]
            delta_Bi[i_sample[i]] += f[i] + self.beta_V * self.items_bias[i_sample[i]]

        #Updates simultaneously
        self.users_embedding = self.users_embedding - self.learning_rate * delta_U
        self.items_embedding = self.items_embedding - self.learning_rate * delta_V
        self.items_bias = self.items_bias - self.learning_rate * delta_Bi

    #np.sum(),axis=1 means the sum of one line
    def get_all_logits(self,u,all_logits=None):
        all_items=np.arange(self.itemNum)
        all_logits=np.sum(np.multiply(self.users_embedding[u],self.items_embedding[all_items]),1)+self.items_bias[all_items]
        return  all_logits

    def get_reward(self,u,sample):
        logits=np.array([0.]*self.itemNum)
        logits[sample] = np.sum(np.multiply(self.users_embedding[u],self.items_embedding[sample]),1)+self.items_bias[sample]
        logits[sample] = 2 * ( 1.0/(1.0+np.exp(-logits[sample])) - 0.5)
        return logits

    # for model performance test
    def get_some_logits(self, u, sample=[]):
        logits=np.array([0.]*self.itemNum)
        logits[sample] = np.sum(np.multiply(self.users_embedding[u],self.items_embedding[sample]),1)+self.items_bias[sample]
        return logits

    def get_user_all_rating(self, user):
        logit = np.sum(np.multiply(self.users_embedding[user], self.items_embedding), 1) + self.items_bias
        return logit
        
    def save_model(self):
        with open('./ml-100k/dis_model','a')as fout:
            for u in range(self.userNum):
                fout.write(str(self.users_embedding[u])+'\n')
            for i in range(self.itemNum):
                fout.write(str(self.items_embedding[i])+'\n')
            for i in range(self.itemNum):
                fout.write(str(self.items_bias[i])+' ')
            fout.write('\n')
            fout.write('\n')
            fout.flush()
            fout.close()
