import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.stats import norm,multivariate_normal

import warnings

x = print()
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# from random import shuffle
from sklearn.utils import shuffle

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import argparse
from sklearn.utils import shuffle
from ope_v3 import *


class WEIGHT_DATA_SET(data.Dataset):
    '''
    dataset class with instance weight
    '''

    def __init__(self, data, reward, weights):
        '''
        weights are same dimensional witi original data
        '''
        self.data = torch.tensor(data)
        self.reward = torch.tensor(reward)
        self.weights = torch.tensor(weights)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.reward[index]
        weight = self.weights[index]
        return img, target, weight

    def __len__(self):
        return len(self.data)
def load_dataset(data):
    if data == 'opt':
        data = pd.read_csv('dataset/optdigits.csv', header=None)
        data = shuffle(data)
        label = np.array(data[64]) - 1
        xy = np.array(data.drop([64], axis=1))
        n_class = 10
        train_size_list = [500,1000,2000, int(len(xy) * 0.7)]

        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]

    elif data == 'satimage':
        data, labels = load_svmlight_file('dataset/satimage.scale')
        data = data.toarray()
        labels = np.array(labels, np.int64)
        index = np.random.permutation(len(data))
        xy = data[index]
        label = labels[index]
        label = label - 1
        n_class = max(label) + 1
        train_size_list = [500,1000,2000, int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]
    elif data == 'veh':
        data, labels = load_svmlight_file('dataset/vehicle.scale')
        data = data.toarray()
        labels = np.array(labels, np.int64)
        index = np.random.permutation(len(data))
        xy = data[index]
        label = labels[index]
        label = label - 1
        n_class = max(label) + 1
        print(n_class)
        train_size_list = [200, 500,  int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]


    elif data == 'pen':
        data = pd.read_csv('dataset/pendigits.tra', sep=',', header=None)
        data = shuffle(data)

        label = np.array(data[16])
        xy = np.array(data.drop([16], axis=1))
        xy = StandardScaler().fit(xy).transform(xy)
        n_class = 10
        train_size_list = [500, 1000, 2000,5000, int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]

    elif data == 'letter':
        data = pd.read_csv('dataset/letter.csv')
        data = shuffle(data)
        label = np.array(data['0'])
        xy = np.array(data.drop(['0'], axis=1))
        xy = StandardScaler().fit(xy).transform(xy)
        n_class = 26
        train_size_list = [5000,10000, int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]

    elif data == 'glass':
        data = pd.read_csv('dataset/glass.data', sep=',', header=None)
        data = shuffle(data)
        data = data.drop([0], axis=1)
        label = np.array(data[10] - 1, int)
        xy = np.array(data.drop([10], axis=1))
        label[label > 3] = (label[label > 3] - 1)
        xy = StandardScaler().fit(xy).transform(xy)
        n_class = 6

        train_size_list = [50, 100, int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]

    elif data == 'ecoli':
        data = pd.read_csv('dataset/ecoli.data', header=None)
        label = data[7]
        xy = np.array(data.drop([7], axis=1))
        le = LabelEncoder()
        label = le.fit_transform(label)

        index = np.random.permutation(len(data))
        xy = xy[index]
        label = label[index]

        n_class = 8
        train_size_list = [50, 100, 200, int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]


    elif data == 'yeast':
        data = pd.read_csv('dataset/yeast.data')
        y = data.iloc[:, -14:]
        label = np.zeros(len(y), int)
        for i in range(len(y.columns)):
            col = y.columns[i]
            label[y[col] == 1] = i

        xy = np.array(data.iloc[:, 0:-14])
        index = np.random.permutation(len(xy))

        xy = xy[index]
        label = label[index].reshape(len(label))
        n_class = 14
        train_size_list = [100, 200, 500, int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]

    elif data == 'page':
        data = pd.read_csv('dataset/page-blocks.data', header=None)
        split_data = []
        for t in data[0]:
            split_data.append(t.split())
        data = np.array(split_data, float)
        label = np.array(data[:, -1] - 1, int)
        xy = data[:, :-1]

        index = np.random.permutation(len(xy))
        xy = xy[index]
        label = label[index]
        xy = StandardScaler().fit(xy).transform(xy)

        train_size_list = [50,100, 1000, 2000, int(len(xy) * 0.7)]
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]
        n_class = 5



    elif data == 'dataset/mnist':
        path = 'training.pt'
        features_, labels = torch.load(path)
        size = 28
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

        features = torch.zeros((len(features_), size, size))
        for i in range(len(features_)):
            features[i] = trans(features_[i].reshape(1,28,28))


        features = features.numpy().reshape(-1, size * size)[:30000]
        labels = labels.numpy()[:30000]

        xy = features
        label = labels
        n_class = 10
        train_size_list = [int(len(xy) * 0.7)]
        test_size_list = [int(len(xy) * 0.3)]
    return xy,label,n_class,train_size_list,test_size_list

def train(dataname,lr,batch_size):
    lr1 = lr
    lr2 = lr

    xy, label, n_class,train_size_list,size_list = load_dataset(dataname)


    onehot_model = OneHotEncoder(sparse=False).fit(label.reshape((len(label), 1)))

    n_dim = xy.shape[1]

    train_xy = xy[:int(len(xy) * 0.7)]
    test_xy = xy[int(len(xy) * 0.7):]

    train_label = label[:int(len(xy) * 0.7)]
    test_label = label[int(len(xy) * 0.7):]

    pca_model = PCA(n_components=1).fit(xy)
    fc = pca_model.transform(xy)

    train_fc = fc[:int(len(xy) * 0.7)]
    test_fc = fc[int(len(xy) * 0.7):]

    train_size = int(len(xy) * 0.7)

    # box 2

    logging_a = 0.8
    logging_b = 0.3

    eval_a = 1
    eval_b = 0.01

    def get_distribution(fc,a,b,c,test = False):
        m = np.min(fc)
        m_bar = np.mean(fc)
        mean = m + (m_bar-m)/a

        # min + (mean - min)/0.8
        # (mean - min) / 0.3
        std = np.sqrt((m_bar - m)/b)
        pdf = norm(loc = mean,scale = std).pdf(fc)
        if c == 1:
            pdf = np.ones((len(fc),1))
            for i in range(len(pdf)):
                if label[i] == 0:
                    pdf[i] = a
                else:
                    pdf[i] = b
        if test:
            pdf = np.ones((len(fc),1))
            return pdf / sum(pdf), 0, 1
        return pdf/sum(pdf), mean, std


    # # box 3
    # # logging policy, eval_policy
    # logging_policy_interval = [0.5,0.8]
    # eval_policy_interval = [0.8,0.9]
    def generate_dirichlet_policy(alpha):
        clf_policy = LogisticRegression(random_state=1).fit(train_xy, train_label)
        predict_label = clf_policy.predict(xy)
        eval_policy = np.zeros((len(xy), n_class))

        p_max = []
        for i in range(len(xy)):
            tmp = np.zeros((1, n_class))
            #         u = np.random.uniform(-0.05,0.05)
            for t in range(n_class):
                if t == predict_label[i]:
                    tmp[0, t] = 0.9
                else:
                    tmp[0, t] = (1 - 0.9) / (n_class - 1)
            eval_policy[i] = tmp


        from scipy.stats import dirichlet
        alpha = np.ones(n_class)*alpha
        prob = dirichlet.rvs(alpha, size=1, random_state=1)

        policy =np.ones((len(xy),n_class))

        for i in range(len(policy)):
            policy[i] = prob

        return policy, eval_policy

    def generate_target_policy(p):
        # eval_policy = np.ones((len(xy),n_class)) * ( (1-p)/(n_class-1))
        #
        # for i in range(len(xy)):
        #     eval_policy[i,label[i]] = (label[i] + 1) /10
        #
        #
        # return eval_policy
        eval_policy = np.zeros((len(xy),n_class))
        problist = []
        for i in range(n_class):
            problist.append((i+1)/n_class)
        shuffle(problist)
        for i in range(len(xy)):
            prob = problist[label[i]]
            eval_policy[i, :] = (1-prob)/(n_class-1)
            eval_policy[i, label[i]] = prob
        return eval_policy

    def get_logging_policy_classifier(xy, label):
        new_xy = []
        new_label = []
        count = {}
        for i in np.unique(label):
            count[i] = i + 1
        for i in range(len(xy[:train_size])):
            if count[label[i]] != 0:
                new_xy.append(xy[i])
                new_label.append(label[i])
                count[label[i]] -= 1
        clf_policy = LogisticRegression(random_state=0).fit(new_xy, new_label)
        predict_label = clf_policy.predict(xy)
        # for i in range(n_class):
        #     index = label == i
        #     print(i, sum(predict_label[index] == label[index])/len(label[index]))
        return clf_policy

    def generate_policy_soften(alpha, beta,bias_training = False):
        # if bias_training == 1:
        #     clf_policy = get_logging_policy_classifier(xy, label)
        # else:
        clf_policy = LogisticRegression(random_state=0).fit(train_xy[:int(train_size * 0.1)],
                                                            train_label[:int(train_size * 0.1)])
        predict_label = clf_policy.predict(xy)
        logging_policy = np.zeros((len(xy), n_class))

        p_max = []
        for i in range(len(xy)):
            tmp = np.zeros((1, n_class))
            u = np.random.uniform(-0.5, 0.5)
            for t in range(n_class):
                if t == predict_label[i]:
                    tmp[0, t] = alpha + u * beta
                else:
                    tmp[0, t] = (1 - alpha - u * beta) / (n_class - 1)
            logging_policy[i] = tmp

        # clf_policy = LogisticRegression(random_state=1).fit(train_xy, train_label)
        if bias_training == 1:
            clf_policy = get_logging_policy_classifier(xy, label)
        else:
            clf_policy = LogisticRegression(random_state=0).fit(train_xy[:int(train_size * 0.1)],
                                                                train_label[:int(train_size * 0.1)])
        predict_label = clf_policy.predict(xy)
        eval_policy = np.zeros((len(xy), n_class))
        p_max = []
        for i in range(len(xy)):
            tmp = np.zeros((1, n_class))
            #         u = np.random.uniform(-0.05,0.05)
            for t in range(n_class):
                if t == predict_label[i]:
                    tmp[0, t] = 0.9
                else:
                    tmp[0, t] = (1 - 0.9) / (n_class - 1)
            eval_policy[i] = tmp
        return logging_policy, eval_policy
    def generate_tweak_1_policy(rho):
        clf_policy = LogisticRegression(random_state=1).fit(train_xy, train_label)
        predict_label = clf_policy.predict(xy)
        eval_policy = np.zeros((len(xy), n_class))
        p_max = []
        for i in range(len(xy)):
            tmp = np.zeros((1, n_class))
            #         u = np.random.uniform(-0.05,0.05)
            for t in range(n_class):
                if t == predict_label[i]:
                    tmp[0, t] = 0.9
                else:
                    tmp[0, t] = (1 - 0.9) / (n_class - 1)
            eval_policy[i] = tmp
        rho = rho[0]
        uniform = (1-rho)/(n_class-1)

        policy =np.ones((len(xy),n_class)) * uniform
        policy[:,-1] = rho
        return policy, eval_policy


    def generate_policy_adv(alpha, beta):
        clf_policy = LogisticRegression(random_state=0).fit(xy, label)
        predict_label = clf_policy.predict(xy)

        logging_policy = np.zeros((len(xy), n_class))

        p_max = []
        for i in range(len(xy)):
            tmp = np.zeros((1, n_class))
            u = np.random.uniform(-0.5, 0.5)
            for t in range(n_class):
                if t == predict_label[i]:
                    tmp[0, t] = 1 - alpha - u * beta
                else:
                    tmp[0, t] = (alpha + u * beta) / (n_class - 1)
            logging_policy[i] = tmp

        clf_policy = LogisticRegression(random_state=1).fit(xy, label)
        predict_label = clf_policy.predict(xy)
        eval_policy = np.zeros((len(xy), n_class))
        p_max = []
        for i in range(len(xy)):
            tmp = np.zeros((1, n_class))
            #         u = np.random.uniform(-0.05,0.05)
            for t in range(n_class):
                if t == predict_label[i]:
                    tmp[0, t] = 0.9
                else:
                    tmp[0, t] = (1 - 0.9) / (n_class - 1)
            eval_policy[i] = tmp
        return logging_policy, eval_policy


    # sample data
    # just sample index and everything can be refered from index.
    def take_action(policy):
        action = random.choices([t for t in range(n_class)], policy)[0]
        prob = policy[action]
        return action, prob

    def sample_action_batch(p, n=1, items=None):
        s = p.cumsum(axis=1)
        r = np.random.rand(p.shape[0], n, 1)
        q = np.expand_dims(s, 1) >= r
        k = q.argmax(axis=-1)
        if items is not None:
            k = np.asarray(items)[k]
        k = k.reshape(len(k))
        return k


    def sample_logging_data(xy, label, size, pdf, pdf_eval, logging_policy, eval_policy):
        logging_index = np.zeros(size, np.int)
        logging_action = np.zeros(size, np.int)
        logging_reward = np.zeros(size, np.int)
        index_list = [i for i in range(len(xy))]
        # print(logging_policy)
        logging_index = random.choices(index_list[:train_size], weights=pdf[:train_size],k = size)
        logging_policy_sampled = logging_policy[logging_index]
        # print(logging_policy_sampled)
        logging_action = sample_action_batch(logging_policy_sampled)
        # print(logging_action)
        truelabel = label[logging_index]
        logging_reward = np.array(logging_action == truelabel,dtype =int)

        # for i in tqdm(range(size)):
        #
        #     t_index = random.choices(index_list[:train_size], weights=pdf[:train_size])[0]
        #     # t_index = i
        #     logging_index[i] = t_index
        #     action, _ = take_action(logging_policy[t_index])
        #     logging_action[i] = action
        #
        #     logging_reward[i] = int(action == label[t_index])

        return logging_index, logging_action, logging_reward

    # ips on data collected on logging policy
    # dm on data collected on target policy
    # triple on data collected on logging (ips part) + target(dm part) policy

    def sample_eval_data_with_logging_xshift(xy, label, size, pdf, pdf_eval, logging_policy, eval_policy):
        eval_index = np.zeros(size, np.int)
        eval_action = np.zeros(size, np.int)
        eval_reward = np.zeros(size, np.int)
        index_list = [i for i in range(len(xy))]


        eval_index = np.array(random.choices(index_list, weights=pdf[train_size:],k = size)) + train_size
        eval_policy_sampled = logging_policy[eval_index]
        eval_action = sample_action_batch(eval_policy_sampled)
        truelabel = label[eval_index-train_size]
        eval_reward = np.array(eval_action == truelabel,dtype =int)

        #
        # for i in range(size):
        #     t_index = random.choices(index_list, weights=pdf[train_size:])[0]
        #     # t_index = i
        #     t_index = t_index + train_size
        #     eval_index[i] = t_index
        #
        #     action, _ = take_action(logging_policy[t_index])
        #     eval_action[i] = action
        #
        #     eval_reward[i] = int(action == label[t_index - train_size])

        return eval_index, eval_action, eval_reward



    def sample_eval_data_with_target_xshift(xy, label, size, pdf, pdf_eval, logging_policy, eval_policy):
        eval_index = np.zeros(size, np.int)
        eval_action = np.zeros(size, np.int)
        eval_reward = np.zeros(size, np.int)
        index_list = [i for i in range(len(xy))]

        eval_index = np.array(np.arange(size)) + train_size
        eval_policy_sampled = logging_policy[eval_index]
        eval_action = sample_action_batch(eval_policy_sampled)
        truelabel = label[eval_index-train_size]
        eval_reward = np.array(eval_action == truelabel,dtype =int)


        # for i in range(size):
        #     t_index = i
        #     t_index = t_index + train_size
        #     eval_index[i] = t_index
        #
        #     action, _ = take_action(logging_policy[t_index])
        #     eval_action[i] = action
        #
        #     eval_reward[i] = int(action == label[t_index - train_size])

        return eval_index, eval_action, eval_reward
    def p_xshift(train_x,test_x):
        from sklearn.linear_model import LogisticRegression
        X = np.concatenate((train_x,test_x),axis = 0)
    #     X = np.sum(X,axis = 1,keepdims = True)
        Y = np.concatenate((np.ones(len(train_x)),np.zeros(len(test_x))),axis = 0)
        clf = LogisticRegression().fit(X,Y)
        return clf

    def loggingpolicy(x,y):
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit(x,y)
        return clf


    def evaluate_ips(logging_index, logging_action, logging_reward, \
                     eval_index, eval_action, eval_reward, \
                     xy, label, pdf, pdf_eval, logging_policy, eval_policy, GT):
        #     GT = np.mean(eval_reward)
        ips_list = []
        snips_list = []

        ips_r_list = []
        snips_r_list = []



        for k in range(1):
            p_w = my_bound(eval_policy[eval_index, eval_action] / logging_policy[eval_index, eval_action])
            x_w = my_bound(pdf_eval[eval_index]/pdf[eval_index]).reshape(len(eval_index))

            ips = p_w * eval_reward
            ips_loss = (np.mean(ips) - GT) ** 2
            ips_list.append(ips_loss)

            snips = np.sum(p_w * eval_reward) / np.sum(p_w)
            snips_loss = (np.mean(snips) - GT) ** 2
            snips_list.append(snips_loss)

            ips_r = x_w * p_w * eval_reward
            ips_loss = (np.mean(ips_r) - GT) ** 2
            ips_r_list.append(ips_loss)

            snips = np.sum(x_w * p_w * eval_reward) / np.sum(p_w*x_w)
            snips_loss = (np.mean(snips) - GT) ** 2

            snips_r_list.append(snips_loss)
        return np.mean(ips_list), np.mean(snips_list), np.mean(ips_r_list), np.mean(snips_r_list), 0

    # r is xshift weight
    # w is policy shift weight
    def train_robust_regression(logging_index, logging_action, logging_reward, \
                                eval_index, eval_action, eval_reward, \
                                xy, label, pdf, pdf_eval, logging_policy, eval_policy, \
                                dm=False, xshift_known=True, policy_known=True, use_xshift=True):


        feature = xy[logging_index]
        eval_xshift = pdf_eval[logging_index]
        logging_xshift = pdf[logging_index]
        eval_policy = eval_policy[logging_index, logging_action]
        logging_policy = logging_policy[logging_index, logging_action]
        action_onehot = onehot_model.transform(np.reshape(logging_action, (len(logging_action), 1)))

        input_feature = np.concatenate((feature, action_onehot), axis=1)

        train_size = len(input_feature)

        if xshift_known:
            r = logging_xshift / eval_xshift
            r = r.squeeze(1)

        if policy_known:
            w = logging_policy / eval_policy
        if use_xshift:
            w = r * w

        if dm:
            w = torch.ones(len(input_feature))
        # print(w.shape)
        weight_st = my_bound(w)
        weighted_train = WEIGHT_DATA_SET(input_feature, logging_reward, w)

        train_model = Net(n_dim + n_class, 64, n_class)

        validate_size = int(0.1 * train_size)
        validate_loader = data.DataLoader(data.Subset(weighted_train, range(0, validate_size)),
                                          batch_size=batch_size, shuffle=True)
        # 10% validation set
        train_loader = data.DataLoader(data.Subset(weighted_train, range(validate_size, train_size)),
                                       batch_size=batch_size, shuffle=True, )

        train_model, Myy, Myx, _, _, _ = train_validate_test(args, args.lr_robust, 400, "regression", 'cpu', 'False',
                                                             train_model,
                                                             train_loader, None, validate_loader, n_class, 0.000,
                                                             d=n_class + n_dim, testflag=False, lr1=lr1, lr2=lr2)

        return train_model, Myy, Myx


    def predict_regression(model_robust_list, Myy_robust_list, Myx_robust_list, \
                           logging_action, \
                           xy, pdf, pdf_eval, logging_policy, eval_policy, \
                           dm=False, xshift_known=True, policy_known=True, use_xshift=True):
        if xshift_known:
            r = pdf / pdf_eval
            r = r.reshape(len(r))

        if policy_known:
            w = logging_policy / eval_policy
        if use_xshift:
            w = r * w
        if dm:
            w = torch.tensor([1])

        weight_st = my_bound(w)
        if len(xy.shape) == 2:
            logging_action_onehot = np.zeros((len(xy), 1))
            for t in range(len(logging_action_onehot)):
                logging_action_onehot[t] = logging_action
            logging_action_onehot = onehot_model.transform(logging_action_onehot)
            input_feature = np.concatenate((xy, logging_action_onehot), axis=1)
        else:
            logging_action_onehot = np.zeros((1, 1))
            logging_action_onehot[0,0] = logging_action
            logging_action_onehot = onehot_model.transform(logging_action_onehot)
            input_feature = np.concatenate((xy.reshape(1, len(xy)), logging_action_onehot), axis=1)

        output = model_robust_list(torch.tensor(input_feature))

        Myy = Myy_robust_list
        Myx = Myx_robust_list

        meanY, varY = ru.predict_regression(torch.tensor(weight_st), Myy, Myx, output, mean0, var0)
        return meanY.detach().numpy()

    def evaluate_robust_ips_nor_dmr(logging_index, logging_action, logging_reward, \
                            eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy, \
                            eval_index_target_policy, eval_action_target_policy, eval_reward_target_policy, \
                            xy, label, pdf, pdf_eval, logging_policy, eval_policy, \
                            GT, model_robust_list, Myy_robust_list, Myx_robust_list, \
                            ):

        ips_w = my_bound(eval_policy[eval_index_logging_policy, eval_action_logging_policy] \
                         / logging_policy[eval_index_logging_policy, eval_action_logging_policy])
        x_w = my_bound(pdf_eval[eval_index_logging_policy] / pdf[eval_index_logging_policy]).reshape(
            len(eval_index_logging_policy))
        dm = False
        use_xshift = True
        w = ips_w
        sn = np.mean(w)

        # this is for dm part
        tmp_pred = np.zeros((len(eval_index_target_policy), n_class))
        for i in range(n_class):
            tmp_eval = predict_regression(model_robust_list, Myy_robust_list, Myx_robust_list, \
                                          i, xy[eval_index_target_policy], pdf[eval_index_target_policy],
                                          pdf_eval[eval_index_target_policy], \
                                          logging_policy[eval_index_target_policy, i],
                                          eval_policy[eval_index_target_policy, i], \
                                          dm=dm, use_xshift=use_xshift)
            tmp_eval = tmp_eval.reshape(-1, len(eval_index_target_policy))
            tmp_pred[:, i] = tmp_eval
        regression_pred = np.sum(tmp_pred * eval_policy[eval_index_target_policy, :], axis=1)

        reward_ips_part = np.zeros(len(eval_index_logging_policy))
        for i in range(len(eval_index_logging_policy)):
            tmp_eval = predict_regression(model_robust_list, Myy_robust_list, Myx_robust_list, \
                                          eval_action_logging_policy[i], xy[eval_index_logging_policy[i]],
                                          pdf[eval_index_logging_policy[i]],
                                          pdf_eval[eval_index_logging_policy[i]], \
                                          logging_policy[eval_index_logging_policy[i], eval_action_logging_policy[i]],
                                          eval_policy[eval_index_logging_policy[i], eval_action_logging_policy[i]], \
                                          dm=dm, use_xshift=use_xshift)
            reward_ips_part[i] = tmp_eval

        triple = w * (eval_reward_logging_policy - reward_ips_part) + np.sum(
            tmp_pred * eval_policy[eval_index_target_policy, :], axis=1)

        sn_triple = w * (eval_reward_logging_policy - reward_ips_part) / sn + np.sum(
            tmp_pred * eval_policy[eval_index_target_policy, :], axis=1)

        return (np.mean(regression_pred) - GT) ** 2, (np.mean(triple) - GT) ** 2, (np.mean(sn_triple) - GT) ** 2

    def evaluate_robust_all(logging_index, logging_action, logging_reward, \
                            eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy, \
                            eval_index_target_policy, eval_action_target_policy, eval_reward_target_policy, \
                            xy, label, pdf, pdf_eval, logging_policy, eval_policy, \
                            GT, model_robust_list, Myy_robust_list, Myx_robust_list, \
                            dm=False, use_xshift=True):

        ips_w = my_bound(eval_policy[eval_index_logging_policy, eval_action_logging_policy] \
                / logging_policy[eval_index_logging_policy, eval_action_logging_policy])
        x_w = my_bound(pdf_eval[eval_index_logging_policy]/pdf[eval_index_logging_policy]).reshape(len(eval_index_logging_policy))
        if use_xshift:
            w = ips_w * x_w
            sn = np.mean(w)
        else:
            w = ips_w
            sn = np.mean(w)


        # this is for dm part
        tmp_pred = np.zeros((len(eval_index_target_policy), n_class))
        for i in range(n_class):
            tmp_eval = predict_regression(model_robust_list, Myy_robust_list, Myx_robust_list, \
                                          i, xy[eval_index_target_policy], pdf[eval_index_target_policy], pdf_eval[eval_index_target_policy], \
                                          logging_policy[eval_index_target_policy, i], eval_policy[eval_index_target_policy, i], \
                                          dm=dm, use_xshift=use_xshift)
            tmp_eval = tmp_eval.reshape(-1, len(eval_index_target_policy))
            tmp_pred[:, i] = tmp_eval
        regression_pred = np.sum(tmp_pred * eval_policy[eval_index_target_policy, :], axis=1)

        reward_ips_part=np.zeros(len(eval_index_logging_policy))
        for i in range(len(eval_index_logging_policy)):
            tmp_eval = predict_regression(model_robust_list, Myy_robust_list, Myx_robust_list, \
                               eval_action_logging_policy[i], xy[eval_index_logging_policy[i]], pdf[eval_index_logging_policy[i]],
                               pdf_eval[eval_index_logging_policy[i]], \
                               logging_policy[eval_index_logging_policy[i], eval_action_logging_policy[i]], eval_policy[eval_index_logging_policy[i], eval_action_logging_policy[i]], \
                               dm=dm, use_xshift=use_xshift)
            reward_ips_part[i] = tmp_eval


        triple = w * (eval_reward_logging_policy - reward_ips_part) + np.sum(
            tmp_pred * eval_policy[eval_index_target_policy, :], axis=1)

        sn_triple = w * (eval_reward_logging_policy -  reward_ips_part) /sn + np.sum(
            tmp_pred * eval_policy[eval_index_target_policy, :], axis=1)



        return (np.mean(regression_pred) - GT) ** 2, (np.mean(triple) - GT) ** 2, (np.mean(sn_triple) - GT) ** 2

    model_list = [
        'known ips',
        'known snips',
        'known ips-r',
        'known snips-r',

        'known dmrobust',
        'known dmdr',
        'known sndmdr',


        'known robust',
        'known triple',
        'known sntriple',

        'known robust-r',
        'known triple-r',
        'known sntriple-r',

        'known triple-robustr',
        'known sntriple-robustr',





    ]


    def estimate_GT_with_eval_policy(xy, label, eval_policy):
        eval_policy = eval_policy[train_size:]

        xy = xy[train_size:]

        label = label[train_size:]
        result = 0

        for i in range(len(xy)):
            result += eval_policy[i, label[i]]
        return result / len(eval_policy)

    def return_dict_with_list(size_list):
        d = {}
        for i in range(len(size_list)):
            d[size_list[i]] = []
        return d


    # ips
    from tqdm import tqdm

    # logging_a_list = [0.5,0.8,1.2]
    # logging_b_list = [0.1,0.2,0.4]
    # size_list = [100, 200, 500]

    # train_size_list = [1000]
    # logging_policy_list = [[0.95, 0.1], [0.7, 0.2], [0.5, 0.2], [0.4, 0.2], [0.4, 0.3], [0.1, 0.0]]
    # logging_policy_list = [[0.95, 0.1], [0.7, 0.2], [0.5, 0.2], [0.4, 0.2], [0.4, 0.3], [0.1, 0.0],[0.99],[0.95],[0.91]]



    logging_policy_list = [[0.95, 0.1,0], [0.7, 0.1,0],[0.99],[0.95],[0.91]]

    dirichlet_alpha = [[1],[0.1]]
    logging_policy_list += dirichlet_alpha
    target_policy = [1,0.1]
    # xshift_logging = [[1.5,2],[2.0,2],[0.6,2],[1.2,2], [1.5,3],[2.0,3],]
    xshift_logging = [[15,1,0],[12,1,0],[9,1,0],[6,1,0],[4,1,0],[2,1,0],[1.5,3,1],[2,2,1],[1.5,2,1],[0.6,2,1]]


    # xshift_logging += [[1.2,3],[0.6,4],[1.2,4]]

    cc = 0
    frame = {}
    frame['model'] = model_list

    for bandit_train_size in train_size_list:
        for logging_interval in logging_policy_list:
            for tgt in target_policy:
                cc += 1
                # print(cc)
                for xshif_log in xshift_logging:
                    if len(logging_interval)==1:
                        if logging_interval in dirichlet_alpha:
                            logging_policy, eval_policy = generate_dirichlet_policy(logging_interval[0])
                        else:
                            logging_policy, eval_policy = generate_tweak_1_policy(logging_interval)
                    else:
                        logging_policy, eval_policy = generate_policy_soften(logging_interval[0], logging_interval[1],logging_interval[2])

                    ips = return_dict_with_list(size_list)
                    snips = return_dict_with_list(size_list)

                    ips_r = return_dict_with_list(size_list)
                    snips_r = return_dict_with_list(size_list)

                    ips_unknown = return_dict_with_list(size_list)
                    snips_unknown = return_dict_with_list(size_list)
                    ips_r_unknown = return_dict_with_list(size_list)

                    robust = return_dict_with_list(size_list)
                    triple_noxshift = return_dict_with_list(size_list)
                    triple = return_dict_with_list(size_list)
                    sntriple_noxshift = return_dict_with_list(size_list)
                    sntriple = return_dict_with_list(size_list)
                    dm = return_dict_with_list(size_list)
                    dmdr_noxshift = return_dict_with_list(size_list)
                    dmdr = return_dict_with_list(size_list)
                    sndmdr = return_dict_with_list(size_list)

                    robust_r = return_dict_with_list(size_list)
                    triple_r= return_dict_with_list(size_list)
                    sntriple_r= return_dict_with_list(size_list)

                    triple_robustr= return_dict_with_list(size_list)
                    sntriple_robustr= return_dict_with_list(size_list)



                    pdf, mean_logging, std_logging = get_distribution(fc, xshif_log[0], xshif_log[1],xshif_log[1])
                    pdf_eval, mean_eval, std_eval = get_distribution(fc, eval_a, eval_b,1,test=True)

                    for _ in tqdm(range(10)):
                        if tgt != 1:
                            eval_policy = generate_target_policy(tgt)

                        GT = estimate_GT_with_eval_policy(xy, label, eval_policy)

                        logging_index, logging_action, logging_reward = \
                            sample_logging_data(train_xy, train_label, bandit_train_size, pdf, pdf_eval, logging_policy,
                                                eval_policy)

                        eval_index_logging, eval_action_logging, eval_reward_logging = \
                            sample_eval_data_with_logging_xshift(test_xy, test_label, 10, pdf, pdf_eval, logging_policy, eval_policy)

                        model_robust_list, Myy_robust_list, Myx_robust_list = train_robust_regression(logging_index, logging_action,
                                                                                                      logging_reward, \
                                                                                                      eval_index_logging, eval_action_logging,
                                                                                                      eval_reward_logging, \
                                                                                                      xy, label, pdf, pdf_eval,
                                                                                                      logging_policy, eval_policy, \
                                                                                                      dm=False, use_xshift=False)
                        model_dm_robust_list, Myy_dm_robust_list, Myx_dm_robust_list = train_robust_regression(logging_index,
                                                                                                               logging_action,
                                                                                                               logging_reward, \
                                                                                                               eval_index_logging,
                                                                                                               eval_action_logging,
                                                                                                               eval_reward_logging, \
                                                                                                               xy, label, pdf,
                                                                                                               pdf_eval,
                                                                                                               logging_policy,
                                                                                                               eval_policy, \
                                                                                                               dm=True,
                                                                                                               use_xshift=True)
                        model_robust_list_xshift, Myy_robust_list_xshift, Myx_robust_list_xshift = train_robust_regression(logging_index,
                                                                                                      logging_action,
                                                                                                      logging_reward, \
                                                                                                      eval_index_logging,
                                                                                                      eval_action_logging,
                                                                                                      eval_reward_logging, \
                                                                                                      xy, label, pdf,
                                                                                                      pdf_eval,
                                                                                                      logging_policy,
                                                                                                      eval_policy, \
                                                                                                      dm=False,
                                                                                                      use_xshift=True)

                        for size in size_list:
                            eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy = \
                                sample_eval_data_with_logging_xshift(test_xy, test_label, size, pdf, pdf_eval, logging_policy, eval_policy)
                            eval_index_target_policy, eval_action_target_policy, eval_reward_target_policy = \
                                sample_eval_data_with_target_xshift(test_xy, test_label, size, pdf, pdf_eval, logging_policy, eval_policy)
                            res = evaluate_ips(logging_index, logging_action, logging_reward, \
                                               eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy, \
                                               xy, label, pdf, pdf_eval, logging_policy, eval_policy, GT)

                            ips[size].append(res[0])
                            snips[size].append(res[1])

                            ips_r[size].append(res[2])
                            snips_r[size].append(res[3])

                            res = evaluate_robust_all(logging_index, logging_action, logging_reward, \
                                                      eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy, \
                                                      eval_index_target_policy, eval_action_target_policy,eval_reward_target_policy,\
                                                      xy, label, pdf, pdf_eval, logging_policy, eval_policy, GT, \
                                                      model_robust_list, Myy_robust_list, Myx_robust_list, \
                                                      dm=False, use_xshift=False)
                            robust[size].append(res[0])
                            triple[size].append(res[1])
                            sntriple[size].append(res[2])

                            res = evaluate_robust_all(logging_index, logging_action, logging_reward, \
                                                      eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy, \
                                                      eval_index_target_policy, eval_action_target_policy,eval_reward_target_policy,\
                                                      xy, label, pdf, pdf_eval, logging_policy, eval_policy, GT, \
                                                      model_dm_robust_list, Myy_dm_robust_list, Myx_dm_robust_list, \
                                                      dm=True, use_xshift=False)
                            dm[size].append(res[0])
                            dmdr[size].append(res[1])
                            sndmdr[size].append(res[2])

                            res = evaluate_robust_all(logging_index, logging_action, logging_reward, \
                                                      eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy, \
                                                      eval_index_target_policy, eval_action_target_policy,eval_reward_target_policy,\
                                                      xy, label, pdf, pdf_eval, logging_policy, eval_policy, GT, \
                                                      model_robust_list_xshift, Myy_robust_list_xshift, Myx_robust_list_xshift, \
                                                      dm=False, use_xshift=True)
                            robust_r[size].append(res[0])
                            triple_r[size].append(res[1])
                            sntriple_r[size].append(res[2])

                            res = evaluate_robust_ips_nor_dmr(logging_index, logging_action, logging_reward, \
                                                      eval_index_logging_policy, eval_action_logging_policy, eval_reward_logging_policy, \
                                                      eval_index_target_policy, eval_action_target_policy,eval_reward_target_policy,\
                                                      xy, label, pdf, pdf_eval, logging_policy, eval_policy, GT, \
                                                      model_robust_list_xshift, Myy_robust_list_xshift, Myx_robust_list_xshift)
                            triple_robustr[size].append(res[1])
                            sntriple_robustr[size].append(res[2])


                        for k in range(len(size_list)):
                            size = size_list[k]
                            policy_name =  str(logging_interval) + '_' + str(tgt) + '_' + str(xshif_log)
                            result_list = []

                            result_list.append(np.sqrt(np.mean(ips[size])))
                            result_list.append(np.sqrt(np.mean(snips[size])))

                            result_list.append(np.sqrt(np.mean(ips_r[size])))
                            result_list.append(np.sqrt(np.mean(snips_r[size])))

                            result_list.append(np.sqrt(np.mean(dm[size])))
                            result_list.append(np.sqrt(np.mean(dmdr[size])))
                            result_list.append(np.sqrt(np.mean(sndmdr[size])))

                            result_list.append(np.sqrt(np.mean(robust[size])))
                            result_list.append(np.sqrt(np.mean(triple[size])))
                            result_list.append(np.sqrt(np.mean(sntriple[size])))

                            result_list.append(np.sqrt(np.mean(robust_r[size])))
                            result_list.append(np.sqrt(np.mean(triple_r[size])))
                            result_list.append(np.sqrt(np.mean(sntriple_r[size])))

                            result_list.append(np.sqrt(np.mean(triple_robustr[size])))
                            result_list.append(np.sqrt(np.mean(sntriple_robustr[size])))
                            frame[policy_name] = result_list
    frame = pd.DataFrame(frame).round(5)
    return frame


datalist = [
    'veh',
    'glass',
    'ecoli',
    'yeast',
    'satimage',
    # 'opt',
    # 'page',
    # 'pen',


    # 'letter',

]


lrlist = [0.5,1]
batchsize_list = [256]
from tqdm import tqdm
for dataname in datalist:
    for lr in lrlist:
        for batchsize in batchsize_list:
            print(dataname, lr, batchsize)
            frame = train(dataname, lr, batchsize)
            savepath = os.path.join('tune', dataname + '_' + str(lr) + '_' + str(batchsize)) + 'onehone_xshift_v6.csv'
            frame.to_csv(savepath)
            # try:
            #
            #     frame = tqdm(train(dataname,lr,batchsize))
            #     savepath = os.path.join('tune',dataname + '_' + str(lr) + '_' + str(batchsize)) + '.csv'
            #     frame.to_csv(savepath)
            # except Exception as e:
            #     print(e)
            #     continue





