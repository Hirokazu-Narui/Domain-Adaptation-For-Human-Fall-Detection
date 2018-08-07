import numpy as np
import os
from scipy.io import loadmat
import scipy
import sys
import cPickle as pkl
import tensorbayes as tb
from itertools import izip
from args import args
import pickle
import glob

def permute_images(x):
    state = np.random.get_state()
    np.random.seed(0)
    idx = np.random.permutation(32 * 32 * 3)
    np.random.set_state(state)

    x = x.reshape(len(x), -1)[:, idx]
    x = x.reshape(-1, 32, 32, 3)

    return x

def u2t(x):
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    return x * 2 - 1

def create_labeled_data(x, y, seed, npc):
    print "Create labeled data, npc:", npc
    state = np.random.get_state()
    np.random.seed(seed)
    shuffle = np.random.permutation(len(x))
    x, y = x[shuffle], y[shuffle]
    np.random.set_state(state)

    x_l, y_l, i_l = [], [], []
    for k in xrange(10):
        idx = y.argmax(-1) == k
        x_l += [x[idx][:npc]]
        y_l += [y[idx][:npc]]
    x_l = np.concatenate(x_l, axis=0)
    y_l = np.concatenate(y_l, axis=0)
    return x_l, y_l

def data_import(domain, mode):
    if domain == "source":
        folder_name = "../Pickle/norm/non_smooth/{0}/{1}/*W400/*/*.pkl".format(domain, mode)
    else:
        folder_name = "../Pickle/norm/non_smooth/{0}/{1}/*W400/person1/*/*.pkl".format(domain, mode)
    ## Data read from pkl file
    x_fall = x_normal = y_fall = y_normal =  None
    files = []
    files = sorted(glob.glob(folder_name))
    for ff in files:
        print(ff)
        if "fall" in ff:
            x_tmp = pickle.load(open(ff, 'r'))
            y_tmp = np.array([1,0])
            x_tmp = x_tmp[np.newaxis, :, :]
            y_tmp = y_tmp[np.newaxis, :]
            if x_fall is None:
                x_fall = x_tmp
            else:
                x_fall = np.vstack((x_fall,x_tmp))
            if y_fall is None:
                y_fall = y_tmp
            else:
                y_fall = np.vstack((y_fall,y_tmp))
        else:
            x_tmp = pickle.load(open(ff, 'r'))
            y_tmp = np.array([0,1])
            x_tmp = x_tmp[np.newaxis, :, :]
            y_tmp = y_tmp[np.newaxis, :]
            if x_normal is None:
                x_normal = x_tmp
            else:
                x_normal = np.vstack((x_normal,x_tmp))
            if y_normal is None:
                y_normal = y_tmp
            else:
                y_normal = np.vstack((y_normal,y_tmp))
    return x_fall, x_normal, y_fall, y_normal


class Data(object):
    def __init__(self, images, labels=None, labeler=None, cast=False):
        self.images = images
        self.labels = labels
        self.labeler = labeler
        self.cast = cast

    def next_batch(self, bs, rep=False, debug = False):
        if debug == True:
            k1 = np.random.choice(np.where(self.labels[:,0]==1)[0], bs, replace=rep)
            k2 = np.random.choice(np.where(self.labels[:,1]==1)[0], bs, replace=rep)
            k_sum = np.concatenate((k1,k2))
            idx = np.random.choice(k_sum, bs, replace=rep)
        elif args.phase == 0:
            k1 = np.random.choice(np.where(self.labels[:,0]==1)[0], bs, replace=rep)
            k2 = np.random.choice(np.where(self.labels[:,1]==1)[0], bs, replace=rep)
            k_sum = np.concatenate((k1,k2))
            idx = np.random.choice(k_sum, bs, replace=rep)
        else:
            idx = np.random.choice(len(self.images), bs, replace=False)
#        print(self.labels[idx])
#        x = u2t(self.images[idx]) if self.cast else self.images[idx]
        x = self.images[idx] if self.cast else self.images[idx]
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y
        # if self.cast:
        #     return u2t(self.images[idx]), self.labels[idx].astype('float32')
        # else:
        #     return self.images[idx], self.labels[idx]

# class PseudoData(object):
#     def __init__(self, domain, datasets, M):
#         print "Constructing pseudodata"
#         sys.stdout.flush()

#         cast = domain not in {'mnist28', 'mnist32', 'mnistm28', 'mnistm32'}
#         print "Casting:", cast

#         trainx, trainy = self.create_pseudodata(datasets.train, M, cast=cast)
#         testx, testy = self.create_pseudodata(datasets.test, M, cast=cast)

#         self.train = Data(trainx, trainy, cast=cast)
#         self.test = Data(testx, testy, cast=cast)

#     def create_pseudodata(self, data, M, cast=False):
#         # Data statistics
#         H, W, C = data.images.shape[1:]
#         n_class = data.labels.shape[-1]

#         datax = tb.nputils.split(data.images, 100)
#         x, y = [], []

#         for dx in datax:
#             if cast: dx = u2t(dx)
#             x += [dx]
#             y += [M.sess.run(M.prob_y, {M.test_x: dx})]

#         x = np.concatenate(x, axis=0)
#         y = np.concatenate(y, axis=0)
#         return x, y

class PseudoData(object):
    def __init__(self, domain, datasets, M):
        print "Constructing pseudodata"
        sys.stdout.flush()

        cast = domain not in {'mnist28', 'mnist32', 'mnistm28', 'mnistm32'}
        print "Casting:", cast
        labeler = tb.function(M.sess, [M.test_x], M.back_y)

        self.train = Data(datasets.train.images, labeler=labeler, cast=cast)
        self.test = Data(datasets.test.images, labeler=labeler, cast=cast)

class WifiMexico(object):
    def __init__(self, shape=(30, 400, 3), seed=0, npc=None):
        print "Loading WifiMexico"
        sys.stdout.flush()

        # import train_data
        x_fall_train, x_normal_train, y_fall_train, y_normal_train = data_import(domain="target",mode="train")
        print(" fall=", len(x_fall_train), " normal=", len(x_normal_train))

        wifi_x_mexico_train = np.r_[x_fall_train, x_normal_train]
        wifi_y_mexico_train = np.r_[y_fall_train, y_normal_train]

        # import test_data
        x_fall_test, x_normal_test, y_fall_test, y_normal_test = data_import(domain="target",mode="validation")
        print(" fall=", len(x_fall_test), " normal=", len(x_normal_test))

        wifi_x_mexico_test = np.r_[x_fall_test, x_normal_test]
        wifi_y_mexico_test = np.r_[y_fall_test, y_normal_test]

        # data transpose(n,50,90) -> (n,30,50,3)
        ch1, ch2, ch3 = np.split(wifi_x_mexico_train,3,axis=2)
        wifi_x_mexico_train_norm_ch = np.stack((ch1, ch2, ch3))
        wifi_x_mexico_train_norm_ch = np.transpose(wifi_x_mexico_train_norm_ch, (1,3,2,0))

        ch1, ch2, ch3 = np.split(wifi_x_mexico_test,3,axis=2)
        wifi_x_mexico_test_norm_ch = np.stack((ch1, ch2, ch3))
        wifi_x_mexico_test_norm_ch = np.transpose(wifi_x_mexico_test_norm_ch, (1,3,2,0))

        trainx = wifi_x_mexico_train_norm_ch
        trainy = wifi_y_mexico_train
        testx = wifi_x_mexico_test_norm_ch
        testy = wifi_y_mexico_test

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

class WifiStanford(object):
    def __init__(self, shape=(30, 400, 3), seed=0, npc=None):
        print "Loading WifiStanford"
        sys.stdout.flush()

        # import train_data
        x_fall_train, x_normal_train, y_fall_train, y_normal_train = data_import(domain="source",mode="train")

        # Eliminate p(y)
        np.random.seed(0)
        index = np.random.choice(range(len(x_normal_train)), 4000)
        x_normal_train, y_normal_train = x_normal_train[index], y_normal_train[index]

        print(" fall=", len(x_fall_train), " normal=", len(x_normal_train))

        wifi_x_stanford_train = np.r_[x_fall_train, x_normal_train]
        wifi_y_stanford_train = np.r_[y_fall_train, y_normal_train]

        # import test_data
        x_fall_test, x_normal_test, y_fall_test, y_normal_test = data_import(domain="source",mode="validation")

        # Eliminate p(y)
        np.random.seed(0)
        index = np.random.choice(range(len(x_normal_test)), 1500)
        x_normal_test, y_normal_test = x_normal_test[index], y_normal_test[index]

        print(" fall=", len(x_fall_test), " normal=", len(x_normal_test))


        wifi_x_stanford_test = np.r_[x_fall_test, x_normal_test]
        wifi_y_stanford_test = np.r_[y_fall_test, y_normal_test]

        # data transpose(n,50,90) -> (n,30,50,3)
        ch1, ch2, ch3 = np.split(wifi_x_stanford_train,3,axis=2)
        wifi_x_stanford_train_norm_ch = np.stack((ch1, ch2, ch3))
        wifi_x_stanford_train_norm_ch = np.transpose(wifi_x_stanford_train_norm_ch, (1,3,2,0))

        ch1, ch2, ch3 = np.split(wifi_x_stanford_test,3,axis=2)
        wifi_x_stanford_test_norm_ch = np.stack((ch1, ch2, ch3))
        wifi_x_stanford_test_norm_ch = np.transpose(wifi_x_stanford_test_norm_ch, (1,3,2,0))

        trainx = wifi_x_stanford_train_norm_ch
        trainy = wifi_y_stanford_train
        testx = wifi_x_stanford_test_norm_ch
        testy = wifi_y_stanford_test

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

class Svhn(object):
    def __init__(self, train='train', shape=(32, 32, 3), seed=0, npc=None, permute=False):
        print "Loading SVHN"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        train = loadmat(os.path.join(path, '{:s}_32x32.mat'.format(train)))
        test = loadmat(os.path.join(path, 'test_32x32.mat'))

        # Change format
        trainx, trainy = self.change_format(train)
        testx, testy = self.change_format(test)

        # Convert to one-hot
        trainy = np.eye(10)[trainy]
        testy = np.eye(10)[testy]

        # Filter via npc if not None
        if npc:
            trainx, trainy = create_labeled_data(trainx, trainy, seed, npc)

        if permute:
            print "Shuffling pixels"
            trainx = permute_images(trainx)
            testx = permute_images(testx)

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

    @staticmethod
    def change_format(mat):
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        return x, y

class Mnist(object):
    def __init__(self, shape=(32, 32, 3), seed=0, npc=None, permute=False):
        print "Loading MNIST"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        data = np.load(os.path.join(path, 'mnist.npz'))
        trainx = np.concatenate((data['x_train'], data['x_valid']), axis=0)
        trainy = np.concatenate((data['y_train'], data['y_valid']))
        trainy = np.eye(10)[trainy].astype('float32')

        testx = data['x_test']
        testy = data['y_test'].astype('int')
        testy = np.eye(10)[testy].astype('float32')

        print "Resizing and norming"
        print "Original MNIST:", (trainx.min(), trainx.max()), trainx.shape
        sys.stdout.flush()
        trainx = self.resize_norm(trainx, shape)
        testx = self.resize_norm(testx, shape)
        print "New MNIST:", (trainx.min(), trainx.max()), trainx.shape

        if permute:
            print "Shuffling pixels"
            trainx = permute_images(trainx)
            testx = permute_images(testx)

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

    @staticmethod
    def resize_norm(x, shape):
        H, W, C = shape
        x = x.reshape(-1, 28, 28)

        if x.shape[1:3] == (H, W):
            resized_x = s2t(x)

        else:
            resized_x = np.empty((len(x), H, W), dtype='float32')

            for i, img in enumerate(x):
                resized_x[i] = u2t(scipy.misc.imresize(img, (H, W)))

        resized_x = resized_x.reshape(-1, H, W, 1)
        resized_x = np.tile(resized_x, (1, 1, 1, C))
        return resized_x


class Mnistm(object):
    def __init__(self, shape=(28, 28, 3), seed=0, npc=None):
        print "Loading MNIST-M"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        data = pkl.load(open(os.path.join(path, 'mnistm_data.pkl')))
        labels = pkl.load(open(os.path.join(path, 'mnistm_labels.pkl')))

        trainx, trainy = data['train'], labels['train']
        validx, validy = data['valid'], labels['valid']
        testx, testy = data['test'], labels['test']
        trainx = np.concatenate((trainx, validx), axis=0)
        trainy = np.concatenate((trainy, validy), axis=0)

        print "Resizing and norming"
        print "Original MNIST-M:", (trainx.min(), trainx.max()), trainx.shape
        sys.stdout.flush()
        trainx = self.resize_norm(trainx, shape)
        testx = self.resize_norm(testx, shape)
        print "New MNIST-M:", (trainx.min(), trainx.max()), trainx.shape

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

    @staticmethod
    def resize_norm(x, shape):
        H, W, C = shape
        x = x.reshape(-1, 28, 28, 3)

        if x.shape[1:3] == (H, W):
            resized_x = u2t(x)

        else:
            resized_x = np.empty((len(x), H, W, 3), dtype='float32')

            for i, img in enumerate(x):
                resized_x[i] = u2t(scipy.misc.imresize(img, (H, W)))

        return resized_x

class SynDigits(object):
    def __init__(self, train='train', shape=(32, 32, 3), seed=0, npc=None):
        print "Loading SynDigits"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        train = loadmat(os.path.join(path, 'synth_train_32x32.mat'))
        test = loadmat(os.path.join(path, 'synth_test_32x32.mat'))

        # Change format
        trainx, trainy = self.change_format(train)
        testx, testy = self.change_format(test)

        # Convert to one-hot
        trainy = np.eye(10)[trainy]
        testy = np.eye(10)[testy]

        # Filter via npc if not None
        if npc:
            trainx, trainy = create_labeled_data(trainx, trainy, seed, npc)

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

    @staticmethod
    def change_format(mat):
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        return x, y

class Gtsrb(object):
    def __init__(self, seed=0):
        print "Loading GTSRB"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        data = loadmat(os.path.join(path, 'gtsrb.mat'))

        # Not really sure what happened here
        data['y'] = data['y'].reshape(-1)

        # Convert to one-hot
        n_class = data['y'].max() + 1
        data['y'] = np.eye(n_class)[data['y']]

        # Create train/test split
        state = np.random.get_state()
        np.random.seed(seed)
        shuffle = np.random.permutation(len(data['X']))
        np.random.set_state(state)

        x = data['X'][shuffle]
        y = data['y'][shuffle]
        n = 31367

        trainx, trainy = x[:n], y[:n]
        testx, testy = x[n:], y[n:]

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

class SynSigns(object):
    def __init__(self, seed=0):
        print "Loading SynSigns"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        data = loadmat(os.path.join(path, 'synsigns.mat'))

        # Not really sure what happened here
        data['y'] = data['y'].reshape(-1)

        # Convert to one-hot
        n_class = data['y'].max() + 1
        data['y'] = np.eye(n_class)[data['y']]

        # Create train/test split
        state = np.random.get_state()
        np.random.seed(seed)
        shuffle = np.random.permutation(len(data['X']))
        np.random.set_state(state)

        x = data['X'][shuffle]
        y = data['y'][shuffle]
        n = 95000

        trainx, trainy = x[:n], y[:n]
        testx, testy = x[n:], y[n:]

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

class Cifar(object):
    def __init__(self):
        print "Loading CIFAR"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        train = loadmat(os.path.join(path, 'cifar_train.mat'))
        test = loadmat(os.path.join(path, 'cifar_test.mat'))

        # Get data
        trainx, trainy = train['X'], train['y']
        testx, testy = test['X'], test['y']

        # Convert to one-hot
        trainy = trainy.reshape(-1)
        testy = testy.reshape(-1)
        trainy = np.eye(9)[trainy]
        testy = np.eye(9)[testy]

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

class Stl(object):
    def __init__(self):
        print "Loading STL"
        sys.stdout.flush()
        path = '/mnt/ilcompf5d0/user/rshu/data'
        train = loadmat(os.path.join(path, 'stl_train.mat'))
        test = loadmat(os.path.join(path, 'stl_test.mat'))

        # Get data
        trainx, trainy = train['X'], train['y']
        testx, testy = test['X'], test['y']

        # Convert to one-hot
        trainy = trainy.reshape(-1)
        testy = testy.reshape(-1)
        trainy = np.eye(9)[trainy]
        testy = np.eye(9)[testy]

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)
