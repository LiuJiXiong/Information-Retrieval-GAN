"""
为Pre-train G模型提供参数users_embedding, items_embeddings, items_bias
"""
import datetime
import sys,os
import pickle as cPickle
import numpy as np
from BASE_Gen import Generative
from time import time
import multiprocessing

cores = multiprocessing.cpu_count()

EMB_DIM=5  #dimensional of latent factor
USER_NUM=943
ITEM_NUM=1683
BATCH_SIZE=16
INIT_DELTA=0.05
DNS_K=5
WORK_DIR='./'
TRAIN_FILE='../ml-100k/movielens-100k-train.txt'
TEST_FILE='../ml-100k/movielens-100k-test.txt'
all_items=set(range(ITEM_NUM))
all_users=set(range(USER_NUM))
user_pos_train = {}
user_pos_test = {}

user_pos_train = {}
with open(WORK_DIR + '../ml-100k/movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
with open(WORK_DIR + '../ml-100k/movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

all_users = list(user_pos_train.keys())

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    # if reverse=true,descend;else ascend
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def simple_tesst_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])



def simple_tesst(model):
    result = np.array([0.] * 6)
    batch_size = 1
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    pool = multiprocessing.Pool(cores)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size
        user_batch_rating=[]
        for u in user_batch:
            user_batch_rating.append(model.get_user_all_rating(u))

        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_tesst_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re
    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret

def init_parameters():
    if len(sys.argv)>1:
        EMB_DIM = sys.argv[1]
        USER_NUM = sys.argv[2]
        ITEM_NUM = sys.argv[3]
        BATCH_SIZE = sys.argv[4]
        INIT_DELTA = sys.argv[5]
        WORK_DIR = sys.argv[6]
        TRAIN_FILE = sys.argv[7]
        TEST_FILE = sys.argv[8]
        all_items = set(range(ITEM_NUM))
    else:
        EMB_DIM = 5
        USER_NUM = 943
        ITEM_NUM = 1683
        BATCH_SIZE = 16
        INIT_DELTA = 0.05
        WORK_DIR = './'
        TRAIN_FILE = '../ml-100k/movielens-100k-train.txt'
        TEST_FILE = '../ml-100k/movielens-100k-test.txt'
        all_items = set(range(ITEM_NUM))

def generate_dns(generator):
    #data=[]
    input_user=[]
    input_pos=[]
    input_neg=[]
    for u in user_pos_train.keys():
        all_logits=[]
        pos=user_pos_train[u]
        item_pos_size=len(user_pos_train[u])
        all_logits=generator.get_all_logits(u, all_logits)
        all_logits=np.array(all_logits)#list2array
        # select some items from dis_observed items
        candiates=list(all_items-set(pos))
        neg=[]

        #negative sampling
        for i in range(item_pos_size):
            choice=np.random.choice(candiates,DNS_K)
            choice_ratings=all_logits[choice]
            neg.append(choice[np.argmax(choice_ratings)])

        for i in range(item_pos_size):
            input_user.append(u)
            input_pos.append(int(pos[i]))
            input_neg.append(int(neg[i]))

    return input_user,input_pos,input_neg

def train(generator):
    flag=flag1=flag2=False
    best=0    
    best_g_users_embedding=generator.users_embedding
    best_g_items_embedding=generator.items_embedding
    best_g_items_bias=generator.items_bias

    for epoch in range(3):
        g_user=[]
        g_pos=[]
        g_neg=[]
        g_user,g_pos,g_neg=generate_dns(generator)
        
        for index in range(len(g_user)):
            u=g_user[index]
            i_pos=g_pos[index]
            i_neg=g_neg[index]

            #update the generator model
            generator.updates(u=u,i_pos=i_pos,i_neg=i_neg)
        
        gen_res = simple_tesst(generator)
        print('Train-Gen: Epoch: %d Prec@3/Prec@5/Prec@10/NDCG@3/NDCG@5/NDCG@10: %.4f %.4f %.4f %.4f %.4f %.4f'%(epoch+1, gen_res[0], gen_res[1], gen_res[2], gen_res[3], gen_res[4], gen_res[5]))

        p_5 = gen_res[1]
        if p_5 > best:
            print('Current-best: Prec@3/Prec@5/Prec@10/NDCG@3/NDCG@5/NDCG@10: %.4f %.4f %.4f %.4f %.4f %.4f'%(gen_res[0], gen_res[1], gen_res[2], gen_res[3], gen_res[4], gen_res[5]))
            best = p_5
            best_g_users_embedding=generator.users_embedding
            best_g_items_embedding=generator.items_embedding
            best_g_items_bias=generator.items_bias
    #free the file description
    generator.save_model(best_g_users_embedding,best_g_items_embedding,best_g_items_bias)

def main():
    init_parameters()
    generator=Generative(itemNum=ITEM_NUM, userNum=USER_NUM, emb_dim=EMB_DIM, lamda=0.1, param=None, initdelta=0.05, learning_rate=0.05)

    train(generator)
    gen_res = simple_tesst(generator)
    print('Final-Gen: Prec@3/Prec@5/Prec@10/NDCG@3/NDCG@5/NDCG@10: %.4f %.4f %.4f %.4f %.4f %.4f'%(gen_res[0], gen_res[1], gen_res[2], gen_res[3], gen_res[4], gen_res[5]))

if __name__ == "__main__":
    main()