import datetime
import sys,os
import pickle as cPickle
import numpy as np
from Gen import Generative
from Discri import Discriminative
from time import time
import multiprocessing

cores = multiprocessing.cpu_count()

EMB_DIM=5  #dimensional of latent factor
USER_NUM=943
ITEM_NUM=1683
BATCH_SIZE=16
INIT_DELTA=0.05
WORK_DIR='./'
TRAIN_FILE='./ml-100k/movielens-100k-train.txt'
TEST_FILE='./ml-100k/movielens-100k-test.txt'
all_items=set(range(ITEM_NUM))
all_users=set(range(USER_NUM))
user_pos_train = {}
user_pos_test = {}

user_pos_train = {}
with open(WORK_DIR + 'ml-100k/movielens-100k-train.txt')as fin:
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
with open(WORK_DIR + 'ml-100k/movielens-100k-test.txt')as fin:
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
"""
def get_precision(generator,p):
    test_users=list(user_pos_test.keys())
    count=0
    for u in test_users:
        i_train=user_pos_train[u]
        i_test=user_pos_test[u] # items occures in testing procedure
        i_for_test=list(all_items-set(i_train)) #candiate items list for selecting
        i_scores=generator.get_some_logits(u,i_for_test)
        i_score_map=[]
        for i in i_for_test:
            i_score_map.append((i_scores[i],i))

        i_score_map=sorted(i_score_map,reverse=True)#descending according to the rating

        index=0
        i=0
        for arr in i_score_map:
            if arr[1] in i_test:
                count+=1
            index+=1
            if index>=p:
                break
    return (count/(len(test_users)*p))

def get_dcg(r,p):
    i=0
    tmp=0.
    for i in range(p):
        if r[i]==1:
            tmp+=1/np.log2(i+2)
        else:
            continue
    return tmp

def get_NDCG(generator,p):
    test_users=list(user_pos_test.keys())# all test users
    tmp=0
    for u in test_users:
        i_train=user_pos_train[u] #occured items in training procedure
        i_test=user_pos_test[u] #occured items in testing procedure
        i_for_test=list(all_items-set(i_train)) #test items for test user u
        i_scores=generator.get_some_logits(u,i_for_test) #ratings of test items
        i_score_map=[]
        for i in i_for_test:
            i_score_map.append((i_scores[i],i))
        i_score_map=sorted(i_score_map,reverse=True)

        r=[]
        for arr in i_score_map:
            if arr[1] in i_test:
                r.append(1)
            else:
                r.append(0)
        dcg_max=get_dcg(sorted(r,reverse=True),p)#descending by the scores
        if(dcg_max==0):
            return 0
        dcg=get_dcg(r,p)
        tmp+=dcg/dcg_max
    return tmp/len(test_users)
"""
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

"""
def test(model):
    prec_3=get_precision(model,p=3)
    prec_5=get_precision(model,p=5)
    prec_10=get_precision(model,p=10)
    ndcg_3=get_NDCG(model,p=3)
    ndcg_5=get_NDCG(model,p=5)
    ndcg_10=get_NDCG(model,p=10)
    return (np.array([prec_3, prec_5, prec_10, ndcg_3, ndcg_5, ndcg_10]))
"""

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
        TRAIN_FILE = 'ml-100k/movielens-100k-train.txt'
        TEST_FILE = 'ml-100k/movielens-100k-test.txt'
        all_items = set(range(ITEM_NUM))

def generate_for_d(generator):
    #data=[]
    input_user=[]
    input_item=[]
    input_label=[]
    for u in user_pos_train.keys():
        all_logits=[]
        pos=user_pos_train[u]
        item_pos_size=len(user_pos_train[u])
        all_logits=generator.get_all_logits(u, all_logits)
        rating=np.array(all_logits)/0.2 #temperature parameter
        exp_rating=np.exp(rating)
        prob=exp_rating/np.sum(exp_rating)

        #negative sampling
        neg=np.random.choice(a=np.arange(ITEM_NUM),size=item_pos_size,p=prob)
        for i in range(len(neg)):
            input_user.append(u)
            input_user.append(u)
            input_item.append(int(pos[i]))
            input_item.append(int(neg[i]))
            input_label.append(1)
            input_label.append(0)

    return input_user,input_item,input_label
    """
            data.append(str(u)+'\t'+str(pos[i])+'\t'+str(neg[i]))
        sum_size+=item_pos_size
    with open('./ml-100k/generate_for_d.txt','w') as fout:
        fout.write('\n'.join(data))
    return sum_size
    """

"""
def get_from_generate_for_d(index,size):
    user=[]
    item=[]
    label=[]
    count=1
    with open('./ml-100k/generate_for_d.txt','r')as fin:
        for line in fin:
            if(count>=index and count<index+size):
                line=line.split()
                user.append(line[0])
                user.append(line[0])
                item.append(line[1])
                item.append(line[2])
                label.append(1)
                label.append(0)
            count+=1
    return user, item, label
"""
def train(generator,discriminator):
    flag=flag1=flag2=False
    best=0
    gen_log = open(WORK_DIR + 'ml-100k/gen_log.txt', 'a')

    for epoch in range(15): #15
        ##dsicriminator epoch
        start2=time()
        g_user=[]
        g_item=[]
        g_label=[]
        for d_epoch in range(100):#100
            if d_epoch % 5 == 0 :
                start1=time()
                g_user,g_item,g_label = generate_for_d(generator)
                end1=time()

                #time spend of generating for d
                if flag==False:
                    flag=True
                    gen_log.write("Generate for d :"+str(end1-start1)+'s. \n')
                    gen_log.flush()

            index=1
            while True:
                if(index>len(g_user)):
                    break

                if(index+BATCH_SIZE<=len(g_user)+1):
                    index_s=index-1
                    index_e=index+2*BATCH_SIZE-1
                    input_user=g_user[index_s:index_e]
                    input_item=g_item[index_s:index_e]
                    input_label=g_label[index_s:index_e]
                    index+=2*BATCH_SIZE
                else:
                    index_s = index - 1
                    index_e = len(g_user)
                    input_user = g_user[index_s:index_e]
                    input_item = g_item[index_s:index_e]
                    input_label = g_label[index_s:index_e]
                    index+=index_e-index+1
                discriminator.updates(input_user,input_item,input_label)
                """
                for i in range(len(input_label)):
                    uid=input_user[i]
                    iid=input_item[i]
                    label=input_label[i]
                    discriminator.updates(int(uid),int(iid),int(label))
                """
        discriminator.save_model()
        # time spend of 100 d_epoch
        end2 = time()
        if flag1 == False:
            flag1 = True
            gen_log.write("100 d_epoch :" + str(end2 - start2) + 's. \n')
            gen_log.flush()

        ##gen epoch
        store_result=[]
        count=0
        for g_epoch in range(50):#50
            for u in user_pos_train.keys():
                #number of positive items rated by user u
                len_pos_items=len(user_pos_train[u])

                lambda_sample=0.2
                g_all_logits=[]

                #gen.get_all_logits(): second parameter-logits of all items;return the sum exp ratings;
                g_all_logits = generator.get_all_logits(u,g_all_logits)
                exp_ratings=np.exp(g_all_logits)
                sum_exp_rating=np.sum(exp_ratings)
                prob=np.array(exp_ratings)/sum_exp_rating
                pn=prob*(1.0 - lambda_sample)
                pn[user_pos_train[u]]+=lambda_sample* (1.0/len_pos_items)
                sample=np.random.choice(a=np.arange(ITEM_NUM),size=2*len_pos_items,p=pn)

                lambda_p = lambda_sample/len_pos_items #\lambda^+
                lambda_1 = np.array([0.]*ITEM_NUM)
                lambda_3 = np.array([0.]*ITEM_NUM)
                lambda_2 = 1 - lambda_sample

                # for sample items, get its lambda parameter in the slides
                lambda_1 = discriminator.get_reward(u,sample)# sample items
                lambda_3[sample] = lambda_1[sample]*(1.0/lambda_2)

                generator.updates(u=u,lambda_p=lambda_p,lambda_1=lambda_1,lambda_2=lambda_2,lambda_3=lambda_3,pos=user_pos_train[u],sample=sample)

            result=simple_tesst(generator)
            print("epoch ", epoch, "gen: ", result)
            buf = '\t'.join([str(x) for x in result])
            gen_log.write(str(epoch) + '\t' + buf + '\n')
            gen_log.flush()
            p_5 = result[1]
            if p_5 > best:
                print('best: ', result)
                best = p_5
                generator.save_model()
    #free the file description
    gen_log.close()

def main():
    start_total = time()
    init_parameters()

    file = open("ml-100k/model_dns_ori.pkl", 'rb')
    param_g = cPickle.load(file, encoding='latin1')
    generator=Generative(itemNum=ITEM_NUM, userNum=USER_NUM, emb_dim=EMB_DIM, lamda=0.0 / BATCH_SIZE, param=param_g, initdelta=0.05, learning_rate=0.001)
    discriminator=Discriminative(itemNum=ITEM_NUM, userNum=USER_NUM, emb_dim=EMB_DIM, lamda=0.2, param=None, initdelta=0.05, learning_rate=0.001)

    print('generator:', str(simple_tesst(generator)))
    print('discriminator:', str(simple_tesst(discriminator)))
    with open(WORK_DIR + 'ml-100k/gen_log.txt', 'w')as fout:
        fout.write("result start :" + str(simple_tesst(generator)) + '. \n')
        fout.close()
    train(generator,discriminator)
    print('final:',str(simple_tesst(generator)))

    end_total = time()
    print('total time spend:',str(end_total-start_total))
    with open(WORK_DIR + 'ml-100k/gen_log.txt', 'a')as fout:
        fout.write("time total :" + str(end_total - start_total) + 's. \n')
        fout.close()

if __name__ == "__main__":
    main()