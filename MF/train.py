import tensorflow as tf
import numpy as np
import os
import sys
import random
import collections
import heapq
import math
import logging
from time import time
import multiprocessing
from scipy.special import softmax, expit
from model import BPRMF, CausalE, IPS_BPRMF, BIASMF
from batch_test import *
from matplotlib import pyplot as plt

cores = multiprocessing.cpu_count() // 2



def compute_2i_regularization_id(items, n_items):
    reg_ids = []
    for x in items:
        if x >= n_items:
            reg_ids.append(x - n_items)
            # reg_ids.append(x)
        elif x < n_items:
            reg_ids.append(x + n_items) # Add number of products to create the 2i representation 

    return reg_ids

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, maxlen, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    tp = 1. / np.log2(np.arange(2, k + 2))
    dcg_max = (tp[:min(maxlen, k)]).sum()
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = dict()
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key = item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    
    return r

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))#P = TP/ (TP+FP)
        recall.append(recall_at_k(r, K, len(user_pos_test)))#R = TP/ (TP+FN)
        ndcg.append(ndcg_at_k(r, K, len(user_pos_test)))
        hit_ratio.append(hit_at_k(r, K))#HR = SIGMA(TP) / SIGMA(test_set)
    # print(hit_ratio)

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data.train_user_list[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data.test_user_list[u]

    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))

	#r为预测命中与否的集合，0未命中，1命中
    r = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, Ks)


def test_one_user_pop(x):
    # user u's ratings for user u
    rating = x[0]
    pop_rating = x[1]
    #uid
    u = x[2]
    #user u's items in the training set
    try:
        training_items = data.train_user_list[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data.test_user_list[u]

    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))

    # pre-rank
    item_score = dict()
    for i in test_items:
        item_score[i] = rating[i]
    pre_max = heapq.nlargest(50, item_score, key = item_score.get)
    # re-rank
    item_score = dict()
    for i in pre_max:
        item_score[i] = pop_rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key = item_score.get)
    #r为预测命中与否的集合，0未命中，1命中
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    return get_performance(user_pos_test, r, Ks)

def valid_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data.train_user_list[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_valid = data.valid_user_list[u]

    all_items = set(range(ITEM_NUM))
    valid_items = list(all_items - set(training_items))

	#r为预测命中与否的集合，0未命中，1命中
    r = ranklist_by_sorted(user_pos_valid, valid_items, rating, Ks)

    return get_performance(user_pos_valid, r, Ks)

def test(sess, model, test_users, batch_test_flag = False, model_type = 'o', valid_set="test", item_pop_test=None, pop_exp = 0):

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}


    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0


    total_rate = np.empty(shape=[0, ITEM_NUM])
    for u_batch_id in range(n_user_batchs):

        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)
                if model_type == 'o':
                    print('o')
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'c':
                    i_rate_batch = sess.run(model.user_const_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'ic':
                    i_rate_batch = sess.run(model.item_const_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'rc':
                    i_rate_batch = sess.run(model.user_rand_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                elif model_type == 'irc':
                    i_rate_batch = sess.run(model.item_rand_ratings, {model.users: user_batch,
                                                                    model.pos_items: item_batch})
                else:
                    print('model type error.')
                    exit()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = list(range(ITEM_NUM))
            if model_type == 'o':
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                total_rate = np.vstack((total_rate, rate_batch))
            elif model_type == 'c':
                rate_batch = sess.run(model.user_const_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'ic':
                rate_batch = sess.run(model.item_const_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rc':
                rate_batch = sess.run(model.user_rand_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'irc':
                rate_batch = sess.run(model.item_rand_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rubi_c':
                rate_batch = sess.run(model.rubi_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type=="direct_minus_c":
                rate_batch = sess.run(model.direct_minus_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rubi_user_c':
                rate_batch = sess.run(model.rubi_ratings_userc, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == 'rubi_both':
                rate_batch = sess.run(model.rubi_ratings_both, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif model_type == "item_pop_test":
                rate_batch = sess.run(model.rubi_ratings_both_poptest, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                # rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                #                                                 model.pos_items: item_batch})
                rate_batch_pop = (rate_batch-np.min(rate_batch))*np.power(item_pop_test, pop_exp)
                # rate_batch_pop = np.ones_like(rate_batch)*np.power(item_pop_test, pop_exp)
            else:
                print('model type error.')
                exit()
        if model_type == "item_pop_test":
            user_batch_rating_uid = zip(rate_batch, rate_batch_pop, user_batch)
            batch_result = pool.map(test_one_user_pop, user_batch_rating_uid)
        else:
            user_batch_rating_uid = zip(rate_batch, user_batch)
            if valid_set=="test":
                batch_result = pool.map(test_one_user, user_batch_rating_uid)

            else:
                batch_result = pool.map(valid_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
    # print(result['hit_ratio'])
    if model_type == 'o':
        print('zk:', total_rate.shape, np.mean(total_rate))
        rate_sum, n = 0, 0
        # print(test_users)
        for user, items in data.train_user_list.items():
            if user not in test_users:
                continue
            idx = 0
            for u_id in test_users:
                if u_id == user:
                    break
                idx += 1
            rate_sum += np.sum(total_rate[idx,items])
            n += len(items)
        print('pos rating:', rate_sum/n*1.0)


    assert count == n_test_users
    pool.close()
    return result

def early_stop(hr, ndcg, recall, precision, cur_epoch, config, stopping_step, flag_step = 10):
    if hr >= config['best_hr']:
        stopping_step = 0
        config['best_hr'] = hr
        config['best_ndcg'] = ndcg
        config['best_recall'] = recall
        config['best_pre'] = precision
        config['best_epoch'] = cur_epoch
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger")
        should_stop = True
    else:
        should_stop = False

    return config, stopping_step, should_stop

if __name__ == '__main__':
    # random.seed(123)
    # tf.set_random_seed(123)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    config = dict()
    config['n_users'] = data.n_users
    config['n_items'] = data.n_items
    model_type = ''
    if args.model == 'mf' or (args.model == 'CausalE' and args.skew == 2):
        model_type = 'mf'
        model = BPRMF(args, config)
        print('MF model.')
    elif args.model == 'CausalE':
        model_type = 'CausalE'
        model = CausalE(args, config)
        print('CausalE model.')
    elif args.model == 'IPSmf':
        model_type = 'IPSmf'
        p_matrix = dict()
        p = []
        for item, users in data.train_item_list.items():
            p_matrix[item] = (len(users)+1)/(data.n_users+1)
        for item in data.items:
            # print(item)
            if item not in p_matrix.keys():
                p_matrix[item] = 1/(data.n_users+1)
            p.append(p_matrix[item])
        # print(p)
        model = IPS_BPRMF(args, config, p)
        print('IPS_MF model.')
    elif args.model == 'biasmf':
        model_type = 'mf'
        model = BIASMF(args, config)
        print('BIASMF model.')

    vars_to_restore = []
    for var in tf.trainable_variables():
        if "item_embedding" in var.name:
            vars_to_restore.append(var)
    saver = tf.train.Saver(max_to_keep=10000)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config = gpu_config)
    sess.run(tf.global_variables_initializer())

    #-----------training without pretrain----------
    if model_type == 'CausalE':
        if args.pretrain == 0:
            t0 = time()
            loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
            config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
            stopping_step = 0

            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, reg_loss, cf_loss = 0., 0., 0., 0.
                n_batch = data.n_train // args.batch_size + 1

                for idx in range(n_batch):
                    users, pos_items, neg_items = data.sample()
                    items = pos_items + neg_items
                    reg_ids = compute_2i_regularization_id(items, ITEM_NUM)
                    _, batch_loss, batch_mf_loss, batch_reg_loss, batch_cf_loss = sess.run([model.opt, model.loss, model.mf_loss, model.reg_loss, model.cf_loss],
                                    feed_dict = {model.users: users,
                                                model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.items: items,
                                                model.reg_items: reg_ids})
                    loss += batch_loss/n_batch
                    mf_loss += batch_mf_loss/n_batch
                    reg_loss += batch_reg_loss/n_batch
                    cf_loss += batch_cf_loss/n_batch

                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
                if (epoch + 1) % args.log_interval != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss, cf_loss)
                        print(perf_str)
                    continue

                t2 = time()
                users_to_test = list(data.test_user_list.keys())
                ret = test(sess, model, users_to_test)
                t3 = time()

                loss_loger.append(loss)
                rec_loger.append(ret['recall'][0])
                pre_loger.append(ret['precision'][0])
                ndcg_loger.append(ret['ndcg'][0])
                hit_loger.append(ret['hit_ratio'][0])

                if args.verbose > 0:
                    perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                            (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, cf_loss, ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                    print(perf_str)

                # *********************************************************
                # save the user & item embeddings for pretraining.
                config, stopping_step, should_stop = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
                if args.save_flag == 1:
                    if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                        os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                    saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))

                if should_stop:
                    print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    break
        else:
            pass
            # print('#load existing models.')
            # model_file = tf.train.latest_checkpoint('{}_{}_checkpoint/wd_{}_lr_{}/'.format(args.model, args.dataset, args.wd, args.lr))
            # saver.restore(sess, model_file)
    # MF
    else:
        # no pretrain
        if args.pretrain == 0:
            t0 = time()
            loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
            config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
            config['best_c_hr'], config['best_c_epoch'], config['best_c'] = 0, 0, 0.0
            stopping_step = 0

            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, reg_loss = 0., 0., 0.
                n_batch = data.n_train // args.batch_size + 1
                

                for idx in range(n_batch):
                    users, pos_items, neg_items = data.sample()
                    if args.train=="normal":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.reg_loss],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
                    elif args.train=="rubi":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two, model.loss_two, model.mf_loss_two, model.reg_loss_two],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
                    elif args.train=="rubibce":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce, model.loss_two_bce, model.mf_loss_two_bce, model.reg_loss_two_bce],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})   
                    elif args.train=="normalbce":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_bce, model.loss_bce, model.mf_loss_bce, model.reg_loss_bce],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
                    elif args.train=="rubibceboth":
                        _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce_both, model.loss_two_bce_both, model.mf_loss_two_bce_both, model.reg_loss_two_bce_both],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})    
                        # print(batch_mf_loss, batch_reg_loss) 
                    # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce, model.loss_two_bce, model.mf_loss_ori, model.mf_loss_item],
                    #             feed_dict = {model.users: users,
                    #                         model.pos_items: pos_items,
                    #                         model.neg_items: neg_items})  
                    # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two, model.loss_two, model.mf_loss_ori_bce, model.mf_loss_item_bce],
                    #                 feed_dict = {model.users: users,
                    #                             model.pos_items: pos_items,
                    #                             model.neg_items: neg_items})
                    # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_two_bce, model.loss_two_bce, model.mf_loss_two_bce, model.reg_loss_two_bce],
                    #                 feed_dict = {model.users: users,
                    #                             model.pos_items: pos_items,
                    #                             model.neg_items: neg_items})      
                    loss += batch_loss/n_batch
                    mf_loss += batch_mf_loss/n_batch
                    reg_loss += batch_reg_loss/n_batch
                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
                if (epoch + 1) % args.log_interval != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss)
                        print(perf_str)
                    continue

                t2 = time()
                if args.valid_set=="test":
                    users_to_test = list(data.test_user_list.keys())
                elif args.valid_set=="valid":
                    users_to_test = list(data.valid_user_list.keys())
                # start testing
                if args.test=="normal" or args.test == 'rubi_user_wise':
                    if args.valid_set=="test":
                        ret = test(sess, model, users_to_test, valid_set="test")
                    elif args.valid_set=="valid":
                        ret = test(sess, model, users_to_test, valid_set="valid")
                    # ret = test(sess, model, users_to_test)
                    t3 = time()
                    loss_loger.append(loss)
                    rec_loger.append(ret['recall'][0])
                    pre_loger.append(ret['precision'][0])
                    ndcg_loger.append(ret['ndcg'][0])
                    hit_loger.append(ret['hit_ratio'][0])
                    if args.verbose > 0:
                        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
                                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                    ret['ndcg'][0], ret['ndcg'][-1])
                        print(perf_str)
                elif args.test=="rubi":
                    print('Epoch %d'%(epoch))
                    test(sess, model, users_to_test, model_type="o", valid_set="test")
                    best_c = 0
                    best_hr = 0
                    best_recall=0
                    best_ndcg=0
                    best_pre=0
                    for c in np.linspace(args.start, args.end, args.step):
                        model.update_c(sess, c)
                        if args.valid_set=="test":
                            if args.train == 'rubibceboth':
                                ret = test(sess, model, users_to_test, model_type="rubi_both", valid_set="test")
                            else:
                                ret = test(sess, model, users_to_test, model_type="rubi_c", valid_set="test")
                        elif args.valid_set=="valid":
                            if args.train == 'rubibceboth':
                                ret = test(sess, model, users_to_test, model_type="rubi_both", valid_set="valid")
                            else:
                                ret = test(sess, model, users_to_test, model_type="rubi_c", valid_set="valid")
                        t3 = time()
                        loss_loger.append(loss)
                        rec_loger.append(ret['recall'][0])
                        pre_loger.append(ret['precision'][0])
                        ndcg_loger.append(ret['ndcg'][0])
                        hit_loger.append(ret['hit_ratio'][0])

                        if ret['hit_ratio'][0] > best_hr:
                            best_hr = ret['hit_ratio'][0]
                            best_recall=ret['recall'][0]
                            best_pre=ret['precision'][0]
                            best_ndcg=ret['ndcg'][0]
                            best_c = c

                        if args.verbose > 0:
                            perf_str = 'c:%.2f [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
                                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                    (c, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                        ret['ndcg'][0], ret['ndcg'][-1])
                            print(perf_str)
                    
                    for c in np.linspace(best_c-1, best_c+1,1):
                        model.update_c(sess, c)
                        if args.valid_set=="test":
                            if args.train == 'rubibceboth':
                                ret = test(sess, model, users_to_test, model_type="rubi_both", valid_set="test")
                            else:
                                ret = test(sess, model, users_to_test, model_type="rubi_c", valid_set="test")
                        elif args.valid_set=="valid":
                            if args.train == 'rubibceboth':
                                ret = test(sess, model, users_to_test, model_type="rubi_both", valid_set="valid")
                            else:
                                ret = test(sess, model, users_to_test, model_type="rubi_c", valid_set="valid")
                        t3 = time()
                        loss_loger.append(loss)
                        rec_loger.append(ret['recall'][0])
                        pre_loger.append(ret['precision'][0])
                        ndcg_loger.append(ret['ndcg'][0])
                        hit_loger.append(ret['hit_ratio'][0])

                        if args.verbose > 0:
                            perf_str = 'c:%.2f [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
                                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                    (c, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                        ret['ndcg'][0], ret['ndcg'][-1])
                            print(perf_str)

                        if ret['hit_ratio'][0] > best_hr:
                            best_hr = ret['hit_ratio'][0]
                            best_recall=ret['recall'][0]
                            best_pre=ret['precision'][0]
                            best_ndcg=ret['ndcg'][0]
                            best_c = c

                        if ret['hit_ratio'][0] > config['best_c_hr']:
                            config['best_c_hr'] = ret['hit_ratio'][0]
                            config['best_c_ndcg'] = ret['ndcg'][0]
                            config['best_c_precision'] = ret['precision'][0]
                            config['best_c_recall'] = ret['recall'][0]
                            config['best_c_epoch'] = epoch
                            config['best_c'] = c

                    ret['hit_ratio'][0]=best_hr
                    ret['recall'][0]=best_recall
                    ret['precision'][0]=best_pre
                    ret['ndcg'][0]=best_ndcg

                # *********************************************************
                # save the user & item embeddings for pretraining.
                config, stopping_step, should_stop = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
                if args.save_flag == 1:
                    if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                        os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                    saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))

                if should_stop and args.early_stop == 1:
                    print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                    logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

                    with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                        print(config['best_epoch'], file = f)

                    if args.test=='rubi':
                        with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_c.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                            print(config['best_c'], file = f)

                    break
        # pretrain
        else:
            print('#load existing models.')
            best_epoch = 9
            model_file = '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, best_epoch)
            saver.restore(sess, model_file)
            pop_item_test = []
            with open("data/ml_10m/pop_predict.txt") as f:
                for line in f:
                    line = line.strip().split()
                    item, pop = int(line[0]), float(line[1])
                    pop_item_test.append(pop)
            pop_item_test = np.array(pop_item_test)

            # prev pop
            # with open("data/ml_10m/item_pop_seq_ori.txt") as f:
            #     item_list = []
            #     pop_item_all = []
            #     for line in f:
            #         line = line.strip().split()
            #         item, pop_list = int(line[0]), [float(x) for x in line[1:]]
            #         item_list.append(item)
            #         pop_item_all.append(pop_list)
            # pop_item_all = np.array(pop_item_all)
            # pop_item_test = pop_item_all[:,11]
            # pop_item_test = np.array([len(data.train_item_list.get(i, [])) for i in range(ITEM_NUM)])
            # plot
            # plt.plot(list(range(ITEM_NUM)), pop_item_test)
            # ground truth
            # pop_item_test = np.array([len(data.test_item_list.get(i, [])) for i in range(ITEM_NUM)])
            # plt.plot(list(range(ITEM_NUM)), pop_item_test)
            # plt.savefig("pop_gt.png")
            # exit()
            users_to_test = list(set(list(data.test_user_list.keys()))&set(list(data.train_user_list.keys())))
            
            # normal perf
            # rubi_both
            for c in np.arange(-10,10,10):
                model.update_c(sess, c)
                ret = test(sess, model, users_to_test, model_type="rubi_both", valid_set="test")
                print("c:",c)
                perf_str = 'normal:  recall=[%.5f, %.5f], ' \
                                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                    (ret['recall'][0], ret['recall'][-1],
                                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                        ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
            for pop_exp in np.arange(0,10,1):
                # pop_exp = np.power(10,pop_exp)
                ret = test(sess, model, users_to_test, item_pop_test = pop_item_test, pop_exp=pop_exp, model_type="item_pop_test", valid_set="test")
                perf_str = 'pop_exp:%f  recall=[%.5f, %.5f], ' \
                                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                    (pop_exp, ret['recall'][0], ret['recall'][-1],
                                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                        ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
                perf_write = 'pop_exp:%f  %.4f %.4f %.4f' % \
                                    (pop_exp, ret['hit_ratio'][0], ret['recall'][0],  ret['ndcg'][0])
                with open("result.txt","a+") as f:
                    f.write(perf_write+"\n")

            
    # [94, 99, 129, 184, 249, 299]

    # print(config['best_c_epoch'], config['best_c_hr'], config['best_c_ndcg'], config['best_c_recall'], config['best_c_precision'])

    if args.early_stop == 0 and args.pretrain == 0 and args.test=='normal':
        print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
        logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

        with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
            print(config['best_epoch'], file = f)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    exit()
    
    best_epoch = 0
    with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'r') as f:
        best_epoch = eval(f.read())
    
    
    model_file = '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, best_epoch)
    saver.restore(sess, model_file)

    if args.out == 1:
        users_to_test = list(data.test_user_list.keys())
        ret = test(sess, model, users_to_test, valid_set="test")



        total_rate = np.empty(shape=[0, ITEM_NUM])
        pool = multiprocessing.Pool(cores)
        test_users = users_to_test
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE

        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        if args.train == 'rubibceboth':
            c=0
            with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/best_c.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'r') as f:
                c=eval(f.read())
            model.update_c(sess, c)

        for u_batch_id in range(n_user_batchs):

            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            item_batch = list(range(ITEM_NUM))

            if args.train == 'normalbce':
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            elif args.train == 'rubibceboth':
                rate_batch = sess.run(model.rubi_ratings_both, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            total_rate = np.vstack((total_rate, rate_batch))

        # print(total_rate[:10])

        total_sorted_id = np.argsort(-total_rate, axis=1)
        count = np.zeros(shape=[ITEM_NUM])
        for user, line in enumerate(total_sorted_id):
            # cutline = line[:10]
            # for item in cutline:
            #     count[item] += 1
            n = 0
            for item in line:
                if user not in data.train_user_list.keys() or item not in data.train_user_list[user]:
                    count[item] += 1
                    n += 1
                if n == 10:
                    break
                
        
        print(sorted_id)
        count = count[sorted_id]
        x = list(range(len(points)+1))
        y = [0,0,0,0,0,0]
        n_y = [0,0,0,0,0,0]
        for n, pop in enumerate(count):
            y[belong[n]] += pop
            n_y[belong[n]] += 1
        for i in range(len(points)+1):
            y[i]/=1.0*n_y[i]
        
        # rate = [1.0*n_y[i]/ITEM_NUM for i in range(len(points)+1)]
        # print(rate)
        # plt.plot(x,y)    
        # plt.yticks(size = 12)
        # plt.xticks(x,('0-10','10-50','50-100','100-200','200-500','500+'), size=12)
        # plt.scatter(x,y,marker='v')
        # plt.ylabel('Average Popularity', size=14)
        # plt.xlabel('Popularity Range', size=14)
        # plt.savefig('./fig.png')
        # plt.cla()



        # -*- coding: utf-8 -*-
        from matplotlib import rc
        rc('mathtext', default='regular')

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.bar(x, rate, color='lightskyblue')
        # ax.set_ylabel('#Items/Total Items', size=14, color='lightskyblue')
        

        # ax2 = ax.twinx()
        # ax2.plot(x, y, color='sandybrown')
        # # ax2.legend(loc=0)
        # ax.grid(axis='y')
        # # ax2.set_xlabel('Popularity Range', size=14)
        # ax2.set_ylabel('Average Popularity', size=14)
        # ax2.scatter(x,y,marker='v', color='sandybrown')
        
        # plt.xticks(x,('0-10','10-50','50-100','100-200','200-500','500+'), size=12)
        # plt.yticks(size = 12)
        # ax.set_xlabel('Item Group', size=14)
        # plt.savefig('./fig.png')
        # plt.cla()
        # plt.clf()
        import os

        import sys
        np.set_printoptions(threshold=sys.maxsize)
        with open('./curve/x.txt', 'w') as f:
            f.write(str(x))
        with open('./curve/itemsorted_id.txt', 'w') as f:
            f.write(str(list(sorted_id)))
        with open('./curve/itembelong.txt', 'w') as f:
            f.write(str(belong))
        with open('./curve/itemrate.txt'.format(args.train), 'w') as f:
            f.write(str(rate))
        with open('./curve/usersorted_id.txt', 'w') as f:
            f.write(str(list(usersorted_id)))
        with open('./curve/userbelong.txt', 'w') as f:
            f.write(str(userbelong))
        with open('./curve/userrate.txt'.format(args.train), 'w') as f:
            f.write(str(userrate))

        if args.dataset == 'addressa':
            with open('./curve/y_{}.txt'.format(args.train), 'w') as f:
                f.write(str(y))

        if args.train == 'rubibceboth':
            sig_yu, sig_yi = sess.run([model.sigmoid_yu, model.sigmoid_yi])

            sig_sum = [0,0,0,0,0,0]
            n_sig = [0,0,0,0,0,0]

            sig_yu = sig_yu[usersorted_id]
            for i, sig in enumerate(sig_yu):
                sig_sum[userbelong[i]] += sig
                n_sig[userbelong[i]] += 1
            
            for i in range(6):
                sig_sum[i]/=1.0*n_sig[i]
            
            with open('./curve/sig_yu_mf.txt', 'w') as f:
                f.write(str(sig_sum))

            sig_sum = [0,0,0,0,0,0]
            n_sig = [0,0,0,0,0,0]

            sig_yi = sig_yi[sorted_id]
            for i, sig in enumerate(sig_yi):
                sig_sum[belong[i]] += sig
                n_sig[belong[i]] += 1
            
            for i in range(6):
                sig_sum[i]/=1.0*n_sig[i]
            
            with open('./curve/sig_yi_mf.txt', 'w') as f:
                f.write(str(sig_sum))

            import matplotlib
            matplotlib.rcParams['figure.figsize'] = [10.5,6.5] # for square canvas
            matplotlib.rcParams['figure.subplot.left'] = 0.2
            matplotlib.rcParams['figure.subplot.bottom'] =0.1
            matplotlib.rcParams['figure.subplot.right'] = 0.8
            matplotlib.rcParams['figure.subplot.top'] = 0.8

            x = np.linspace(0, 60, 41)
            y = []
            for c in x:
                model.update_c(sess, c)
                ret = test(sess, model, users_to_test, model_type="rubi_both", valid_set="test")
                y.append(ret['hit_ratio'][0])
            plt.plot(x, y, color='sandybrown')
            plt.scatter(x, y, color='sandybrown')
            plt.grid(alpha=0.3)
            plt.xlabel('c', size=24, fontweight='bold')
            plt.ylabel('HR@20', size=24, fontweight='bold')
            plt.xticks(size=16, fontweight='bold')
            plt.yticks(size = 16, fontweight='bold')

            plt.savefig('./curve/hr_addressa_causalmf.png')
            plt.cla()



            
        
    if args.valid_set=="valid" or args.valid_set=="test":
        if args.test=="normal" or args.test == 'rubi_user_wise':
            users_to_test = list(data.test_user_list.keys())
            ret = test(sess, model, users_to_test, valid_set="test")
            perf_str = 'Epoch %d:recall=[%.5f, %.5f], ' \
                                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                (best_epoch, ret['recall'][0], ret['recall'][-1],
                                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                    ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
        elif args.test=="rubi":
            for c in np.linspace(args.start, args.end, args.step):
                model.update_c(sess, c)
                users_to_test = list(data.test_user_list.keys())
                if args.train == 'rubibceboth':
                    ret = test(sess, model, users_to_test, model_type="rubi_both", valid_set="test")
                else:
                    ret = test(sess, model, users_to_test, model_type="rubi_c", valid_set="test")
                if args.verbose > 0:
                    perf_str = 'c:%.2f, recall=[%.5f, %.5f], ' \
                            'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                            (c, ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                    print(perf_str)
    # exit()




    if args.test!='rubi_user_wise':
        exit()




    # print('start training c...')
    # t0 = time()
    # loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
    # config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
    # stopping_step = 0


    # if args.train_c=="val":
    #     loss, mf_loss, reg_loss = 0., 0., 0.
    #     n_batch = data.n_valid // args.batch_size + 1
    #     for idx in range(n_batch):
    #         users, pos_items, neg_items = data.sample2()
    #         if args.train == 'normal':
    #             batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2, model.mf_loss2, model.reg_loss2],
    #                                     feed_dict = {model.users: users,
    #                                                 model.pos_items: pos_items,
    #                                                 model.neg_items: neg_items})
    #         elif args.train == 'normalbce':
    #             batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2_bce, model.mf_loss2_bce, model.reg_loss2_bce],
    #                                     feed_dict = {model.users: users,
    #                                                 model.pos_items: pos_items,
    #                                                 model.neg_items: neg_items})

    #         # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.reg_loss],
    #         #                         feed_dict = {model.users: users,
    #         #                                     model.pos_items: pos_items,
    #         #                                     model.neg_items: neg_items})
    #         loss += batch_loss/n_batch
    #         mf_loss += batch_mf_loss/n_batch
    #         reg_loss += batch_reg_loss/n_batch
        
    #     print('%.8f=%.8f+%.8f'%(loss, mf_loss, reg_loss))


    #     for epoch in range(args.epoch):
    #         t1 = time()
    #         loss, mf_loss, reg_loss = 0., 0., 0.
    #         n_batch = data.n_valid // args.batch_size + 1
                    

    #         for idx in range(n_batch):
    #             users, pos_items, neg_items = data.sample2()
    #             # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt2, model.loss2, model.mf_loss2, model.reg_loss2],
    #             #                         feed_dict = {model.users: users,
    #             #                                     model.pos_items: pos_items,
    #             #                                     model.neg_items: neg_items})
    #             if args.train == 'normal':
    #                 _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt3, model.loss2, model.mf_loss2, model.reg_loss2],
    #                                         feed_dict = {model.users: users,
    #                                                     model.pos_items: pos_items,
    #                                                     model.neg_items: neg_items})
    #             elif args.train == 'normalbce':
    #                 _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt3_bce, model.loss2_bce, model.mf_loss2_bce, model.reg_loss2_bce],
    #                                         feed_dict = {model.users: users,
    #                                                     model.pos_items: pos_items,
    #                                                     model.neg_items: neg_items})
    #             loss += batch_loss/n_batch
    #             mf_loss += batch_mf_loss/n_batch
    #             reg_loss += batch_reg_loss/n_batch
    #         if np.isnan(loss) == True:
    #             print('ERROR: loss is nan.')
    #             sys.exit()

    #                 # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
    #         if (epoch + 1) % args.log_interval != 0:
    #             if args.verbose > 0 and epoch % args.verbose == 0:
    #                 perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss)
    #                 print(perf_str)
    #             continue

    #         t2 = time()
    #         users_to_test = list(data.test_user_list.keys())
    #         ret = test(sess, model, users_to_test, model_type = 'c')
    #         t3 = time()

    #         loss_loger.append(loss)
    #         rec_loger.append(ret['recall'][0])
    #         pre_loger.append(ret['precision'][0])
    #         ndcg_loger.append(ret['ndcg'][0])
    #         hit_loger.append(ret['hit_ratio'][0])

    #         if args.verbose > 0:
    #             perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
    #                             'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
    #                             (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
    #                                 ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
    #                                 ret['ndcg'][0], ret['ndcg'][-1])
    #             print(perf_str)

    #                 # *********************************************************
    #                 # save the user & item embeddings for pretraining.
    #         config, stopping_step, should_stop = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
    #         if args.save_flag == 1:
    #             if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
    #                 os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
    #             saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_final_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))

    #         if should_stop:
    #             print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
    #             logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

    #             with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/final_best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
    #                 print(config['best_epoch'], file = f)

    #             break
    # elif args.train_c=="test":
    #     loss, mf_loss, reg_loss = 0., 0., 0.
    #     n_batch = data.n_test // args.batch_size + 1
    #     for idx in range(n_batch):
    #         users, pos_items, neg_items = data.sample_test()
    #         if args.train == 'normal':
    #             batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2, model.mf_loss2, model.reg_loss2],
    #                                     feed_dict = {model.users: users,
    #                                                 model.pos_items: pos_items,
    #                                                 model.neg_items: neg_items})
    #         elif args.train == 'normalbce':
    #             batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2_bce, model.mf_loss2_bce, model.reg_loss2_bce],
    #                                     feed_dict = {model.users: users,
    #                                                 model.pos_items: pos_items,
    #                                                 model.neg_items: neg_items})
    #         loss += batch_loss/n_batch
    #         mf_loss += batch_mf_loss/n_batch
    #         reg_loss += batch_reg_loss/n_batch
    #     print('%.8f=%.8f+%.8f'%(loss, mf_loss, reg_loss))
    #     for epoch in range(args.epoch):
    #         t1 = time()
    #         loss, mf_loss, reg_loss = 0., 0., 0.
    #         n_batch = data.n_test // args.batch_size + 1
                    

    #         for idx in range(n_batch):
    #             users, pos_items, neg_items = data.sample_test()
    #             # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt2, model.loss2, model.mf_loss2, model.reg_loss2],
    #             #                         feed_dict = {model.users: users,
    #             #                                     model.pos_items: pos_items,
    #             #                                     model.neg_items: neg_items})
    #             _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt3, model.loss2, model.mf_loss2, model.reg_loss2],
    #                                     feed_dict = {model.users: users,
    #                                                 model.pos_items: pos_items,
    #                                                 model.neg_items: neg_items})
    #             loss += batch_loss/n_batch
    #             mf_loss += batch_mf_loss/n_batch
    #             reg_loss += batch_reg_loss/n_batch
    #         if np.isnan(loss) == True:
    #             print('ERROR: loss is nan.')
    #             sys.exit()

    #                 # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
    #         if (epoch + 1) % args.log_interval != 0:
    #             if args.verbose > 0 and epoch % args.verbose == 0:
    #                 perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss)
    #                 print(perf_str)
    #             continue

    #         t2 = time()
    #         users_to_test = list(data.test_user_list.keys())
    #         ret = test(sess, model, users_to_test, model_type = 'c')
    #         t3 = time()

    #         loss_loger.append(loss)
    #         rec_loger.append(ret['recall'][0])
    #         pre_loger.append(ret['precision'][0])
    #         ndcg_loger.append(ret['ndcg'][0])
    #         hit_loger.append(ret['hit_ratio'][0])

    #         if args.verbose > 0:
    #             perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
    #                             'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
    #                             (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
    #                                 ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
    #                                 ret['ndcg'][0], ret['ndcg'][-1])
    #             print(perf_str)

    #                 # *********************************************************
    #                 # save the user & item embeddings for pretraining.
    #         config, stopping_step, should_stop = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
    #         if args.save_flag == 1:
    #             if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
    #                 os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
    #             saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_final_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))

    #         if should_stop:
    #             print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
    #             logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

    #             with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/final_best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
    #                 print(config['best_epoch'], file = f)

    #             break
        
    
    # with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/final_best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'r') as f:
    #     best_epoch = eval(f.read())
    # model_file = '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_final_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, best_epoch)
    # saver.restore(sess, model_file)

    # # c = sess.run(model.const_embedding)
    # # np.set_printoptions(threshold=np.inf)
    # # print(c)


    # loss, mf_loss, reg_loss = 0., 0., 0.
    # n_batch = data.n_train // args.batch_size + 1
                

    # for idx in range(n_batch):
    #     users, pos_items, neg_items = data.sample()
    #             #     print(sess.run([model.temp1, model.temp2, model.temp3],
    #             #                     feed_dict = {model.users: users,
    #             #                                 model.pos_items: pos_items,
    #             #                                 model.neg_items: neg_items}))
    #             #     if idx == 2:
    #             #         exit()
    #     batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2, model.mf_loss2, model.reg_loss2],
    #                                 feed_dict = {model.users: users,
    #                                             model.pos_items: pos_items,
    #                                             model.neg_items: neg_items})
    #     loss += batch_loss/n_batch
    #     mf_loss += batch_mf_loss/n_batch
    #     reg_loss += batch_reg_loss/n_batch
    
    # print('%.8f=%.8f+%.8f'%(loss, mf_loss, reg_loss))


    print('start training c...')
    t0 = time()
    loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
    config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
    stopping_step = 0


    if args.train_c=="val":
        loss, mf_loss, reg_loss = 0., 0., 0.
        n_batch = data.n_valid // args.batch_size + 1
        for idx in range(n_batch):
            users, pos_items, neg_items = data.sample2()
            
            batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss_userc_bce, model.mf_loss_userc_bce, model.reg_loss_userc_bce],
                                    feed_dict = {model.users: users,
                                                model.pos_items: pos_items,
                                                model.neg_items: neg_items})

            # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.reg_loss],
            #                         feed_dict = {model.users: users,
            #                                     model.pos_items: pos_items,
            #                                     model.neg_items: neg_items})
            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            reg_loss += batch_reg_loss/n_batch
        
        print('%.8f=%.8f+%.8f'%(loss, mf_loss, reg_loss))


        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, reg_loss = 0., 0., 0.
            n_batch = data.n_valid // args.batch_size + 1
                    

            for idx in range(n_batch):
                users, pos_items, neg_items = data.sample2()
                # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt2, model.loss2, model.mf_loss2, model.reg_loss2],
                #                         feed_dict = {model.users: users,
                #                                     model.pos_items: pos_items,
                #                                     model.neg_items: neg_items})
                _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_userc_bce, model.loss_userc_bce, model.mf_loss_userc_bce, model.reg_loss_userc_bce],
                                    feed_dict = {model.users: users,
                                                model.pos_items: pos_items,
                                                model.neg_items: neg_items})
                loss += batch_loss/n_batch
                mf_loss += batch_mf_loss/n_batch
                reg_loss += batch_reg_loss/n_batch
            if np.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

                    # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % args.log_interval != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss)
                    print(perf_str)
                continue

            t2 = time()
            users_to_test = list(data.test_user_list.keys())
            ret = test(sess, model, users_to_test, model_type = 'rubi_user_c')
            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'][0])
            pre_loger.append(ret['precision'][0])
            ndcg_loger.append(ret['ndcg'][0])
            hit_loger.append(ret['hit_ratio'][0])

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
                                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                    ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

                    # *********************************************************
                    # save the user & item embeddings for pretraining.
            config, stopping_step, should_stop = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
            if args.save_flag == 1:
                if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                    os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_final_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))

            if should_stop:
                print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

                with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/final_best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                    print(config['best_epoch'], file = f)

                break
    elif args.train_c=="test":
        loss, mf_loss, reg_loss = 0., 0., 0.
        n_batch = data.n_test // args.batch_size + 1
        for idx in range(n_batch):
            users, pos_items, neg_items = data.sample_test()
            if args.train == 'normal':
                batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2, model.mf_loss2, model.reg_loss2],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
            elif args.train == 'normalbce':
                batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2_bce, model.mf_loss2_bce, model.reg_loss2_bce],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            reg_loss += batch_reg_loss/n_batch
        print('%.8f=%.8f+%.8f'%(loss, mf_loss, reg_loss))
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, reg_loss = 0., 0., 0.
            n_batch = data.n_test // args.batch_size + 1
                    

            for idx in range(n_batch):
                users, pos_items, neg_items = data.sample_test()
                # _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt2, model.loss2, model.mf_loss2, model.reg_loss2],
                #                         feed_dict = {model.users: users,
                #                                     model.pos_items: pos_items,
                #                                     model.neg_items: neg_items})
                _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt3, model.loss2, model.mf_loss2, model.reg_loss2],
                                        feed_dict = {model.users: users,
                                                    model.pos_items: pos_items,
                                                    model.neg_items: neg_items})
                loss += batch_loss/n_batch
                mf_loss += batch_mf_loss/n_batch
                reg_loss += batch_reg_loss/n_batch
            if np.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

                    # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % args.log_interval != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss)
                    print(perf_str)
                continue

            t2 = time()
            users_to_test = list(data.test_user_list.keys())
            ret = test(sess, model, users_to_test, model_type = 'c')
            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'][0])
            pre_loger.append(ret['precision'][0])
            ndcg_loger.append(ret['ndcg'][0])
            hit_loger.append(ret['hit_ratio'][0])

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.8f=%.8f + %.8f], recall=[%.5f, %.5f], ' \
                                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                                (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                    ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

                    # *********************************************************
                    # save the user & item embeddings for pretraining.
            config, stopping_step, should_stop = early_stop(ret['hit_ratio'][0], ret['ndcg'][0], ret['recall'][0], ret['precision'][0], epoch, config, stopping_step)
            if args.save_flag == 1:
                if os.path.exists('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID)) == False:
                    os.makedirs('{}_{}_checkpoint/wd_{}_lr_{}_{}/'.format(args.model, args.dataset, args.wd, args.lr, args.saveID))
                saver.save(sess, '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_final_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, epoch))

            if should_stop:
                print("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))
                logging.info("{} dataset best epoch{}: hr:{} ndcg:{} recall:{} precision:{}".format(args.dataset, config['best_epoch'],config['best_hr'],config['best_ndcg'], config['best_recall'], config['best_pre']))

                with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/final_best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'w') as f:
                    print(config['best_epoch'], file = f)

                break
        
    
    with open('{}_{}_checkpoint/wd_{}_lr_{}_{}/final_best_epoch.txt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID),'r') as f:
        best_epoch = eval(f.read())
    model_file = '{}_{}_checkpoint/wd_{}_lr_{}_{}/{}_final_ckpt.ckpt'.format(args.model, args.dataset, args.wd, args.lr, args.saveID, best_epoch)
    saver.restore(sess, model_file)

    # c = sess.run(model.const_embedding)
    # np.set_printoptions(threshold=np.inf)
    # print(c)

    exit()

    loss, mf_loss, reg_loss = 0., 0., 0.
    n_batch = data.n_train // args.batch_size + 1
                

    for idx in range(n_batch):
        users, pos_items, neg_items = data.sample()
                #     print(sess.run([model.temp1, model.temp2, model.temp3],
                #                     feed_dict = {model.users: users,
                #                                 model.pos_items: pos_items,
                #                                 model.neg_items: neg_items}))
                #     if idx == 2:
                #         exit()
        batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.loss2, model.mf_loss2, model.reg_loss2],
                                    feed_dict = {model.users: users,
                                                model.pos_items: pos_items,
                                                model.neg_items: neg_items})
        loss += batch_loss/n_batch
        mf_loss += batch_mf_loss/n_batch
        reg_loss += batch_reg_loss/n_batch
    
    print('%.8f=%.8f+%.8f'%(loss, mf_loss, reg_loss))