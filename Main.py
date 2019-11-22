import datetime, sys, os, logging
import pickle as cPickle
import numpy as np
from Gen import Generative
from Dis import Discriminative
from time import time
import multiprocessing

cores = multiprocessing.cpu_count()

EMB_DIM = 5  # dimensional of latent factor
USER_NUM = 943
ITEM_NUM = 1683
BATCH_SIZE = 16
INIT_DELTA = 0.05
WORK_DIR = './'
TRAIN_FILE = './ml-100k/movielens-100k-train.txt'
TEST_FILE = './ml-100k/movielens-100k-test.txt'
all_items = set(range(ITEM_NUM))
all_users = set(range(USER_NUM))
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
        user_batch_rating = []
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
    if len(sys.argv) > 1:
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
    # data=[]
    input_user = []
    input_item = []
    input_label = []
    for u in user_pos_train.keys():
        all_logits = []
        pos = user_pos_train[u]
        item_pos_size = len(user_pos_train[u])
        all_logits = generator.get_all_logits(u, all_logits)
        rating = np.array(all_logits) / 0.2  # temperature parameter
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        # negative sampling
        neg = np.random.choice(a=np.arange(ITEM_NUM), size=item_pos_size, p=prob)
        for i in range(len(neg)):
            input_user.append(u)
            input_user.append(u)
            input_item.append(int(pos[i]))
            input_item.append(int(neg[i]))
            input_label.append(1)
            input_label.append(0)

    return input_user, input_item, input_label

def train(generator, discriminator):
    d_epochs, g_epochs, best = 100, 50, 0
    best_g_users_embedding = generator.users_embedding
    best_g_items_embedding = generator.items_embedding
    best_g_items_bias = generator.items_bias

    for epoch in range(15): 
        start2 = time()
        g_user = []
        g_item = []
        g_label = []
        for d_epoch in range(d_epochs):  # 100
            if d_epoch % 5 == 0:
                g_user, g_item, g_label = generate_for_d(generator)

            index = 1
            while True:
                if (index > len(g_user)):
                    break

                if (index + BATCH_SIZE <= len(g_user) + 1):
                    index_s = index - 1
                    index_e = index + 2 * BATCH_SIZE - 1
                    input_user = g_user[index_s:index_e]
                    input_item = g_item[index_s:index_e]
                    input_label = g_label[index_s:index_e]
                    index += 2 * BATCH_SIZE
                else:
                    index_s = index - 1
                    index_e = len(g_user)
                    input_user = g_user[index_s:index_e]
                    input_item = g_item[index_s:index_e]
                    input_label = g_label[index_s:index_e]
                    index += index_e - index + 1
                discriminator.updates(input_user, input_item, input_label)

        # Store the discriminative model
        if epoch == d_epochs - 1:
            discriminator.save_model()

        ##gen epoch
        store_result = []
        count = 0
        for g_epoch in range(g_epochs):  # 50
            for u in user_pos_train.keys():
                # number of positive items rated by user u
                len_pos_items = len(user_pos_train[u])

                lambda_sample = 0.2
                g_all_logits = []

                # gen.get_all_logits(): second parameter-logits of all items;return the sum exp ratings;
                g_all_logits = generator.get_all_logits(u, g_all_logits)
                exp_ratings = np.exp(g_all_logits)
                sum_exp_rating = np.sum(exp_ratings)
                prob = np.array(exp_ratings) / sum_exp_rating
                pn = prob * (1.0 - lambda_sample)
                pn[user_pos_train[u]] += lambda_sample * (1.0 / len_pos_items)
                sample = np.random.choice(a=np.arange(ITEM_NUM), size=2 * len_pos_items, p=pn)

                lambda_p = lambda_sample / len_pos_items  # \lambda^+
                lambda_1 = np.array([0.] * ITEM_NUM)
                lambda_3 = np.array([0.] * ITEM_NUM)
                lambda_2 = 1 - lambda_sample

                # for sample items, get its lambda parameter in the slides
                lambda_1 = discriminator.get_reward(u, sample)  # sample items
                lambda_3[sample] = lambda_1[sample] * (1.0 / lambda_2)

                generator.updates(u=u, lambda_p=lambda_p, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3,
                                  pos=user_pos_train[u], sample=sample)

            gen_res = simple_tesst(generator)
            logging.info('Train-Gen: Epoch: %d Prec@3/Prec@5/Prec@10/NDCG@3/NDCG@5/NDCG@10: %.4f %.4f %.4f %.4f %.4f %.4f'%(epoch, gen_res[0], gen_res[1], gen_res[2], gen_res[3], gen_res[4], gen_res[5]))
            p_5 = gen_res[1]
            if p_5 > best:
                best = p_5
                # store the params of gen with the best precision@5.
                best_g_users_embedding = generator.users_embedding
                best_g_items_embedding = generator.items_embedding
                best_g_items_bias = generator.items_bias
    generator.save_model(best_g_users_embedding, best_g_items_embedding, best_g_items_bias)


def main():
    log_dir = './Log/'
    logging.basicConfig(filename=os.path.join(log_dir,"gen_log.txt"), level=logging.INFO)

    st_time = time()
    init_parameters()

    # load pre_train model for Gen
    file = open("ml-100k/model_dns_ori.pkl", 'rb')
    param_g = cPickle.load(file, encoding='latin1')
    generator = Generative(itemNum=ITEM_NUM, userNum=USER_NUM, emb_dim=EMB_DIM, lamda=0.0 / BATCH_SIZE, param=param_g, initdelta=0.05, learning_rate=0.001)
    discriminator = Discriminative(itemNum=ITEM_NUM, userNum=USER_NUM, emb_dim=EMB_DIM, lamda=0.2, param=None, initdelta=0.05, learning_rate=0.001)
    gen_res = simple_tesst(generator)
    dis_res = simple_tesst(discriminator)
    logging.info('Start-Gen: Prec@3/Prec@5/Prec@10/NDCG@3/NDCG@5/NDCG@10: %.4f %.4f %.4f %.4f %.4f %.4f'%(gen_res[0], gen_res[1], gen_res[2], gen_res[3], gen_res[4], gen_res[5]))
    logging.info('Start-Dis: Prec@3/Prec@5/Prec@10/NDCG@3/NDCG@5/NDCG@10: %.4f %.4f %.4f %.4f %.4f %.4f'%(dis_res[0], dis_res[1], dis_res[2], dis_res[3], dis_res[4], dis_res[5]))

    train(generator, discriminator)
    gen_res = simple_tesst(generator)
    logging.info('Final-Gen: Prec@3/Prec@5/Prec@10/NDCG@3/NDCG@5/NDCG@10: %.4f %.4f %.4f %.4f %.4f %.4f'%(gen_res[0], gen_res[1], gen_res[2], gen_res[3], gen_res[4], gen_res[5]))
    et_time = time()
    logging.info('Time spend: [%.2f].'%(et_time - st_time))

if __name__ == "__main__":
    main()