from args import args
import tensorflow as tf
import shutil
import os
import datasets
import numpy as np

def u2t(x):
    return x.astype('float32') / 255 * 2 - 1

def delete_existing(path):
    if args.run < 999:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)

    else:
        if os.path.exists(path):
            shutil.rmtree(path)

def save_accuracy(M, fn_acc_key, tag, dataloader,
                  train_writer=None, step=None, print_list=None,
                  full=True):
    fn_acc = getattr(M, fn_acc_key, None)
    if fn_acc:
        acc, summary = exact_accuracy(fn_acc, tag, dataloader, full)
        train_writer.add_summary(summary, step + 1)
        print_list += [os.path.basename(tag), acc]

def exact_accuracy(fn_acc, tag, dataloader, full=True):
    # Fixed shuffling scheme
    state = np.random.get_state()
    np.random.seed(0)
    shuffle = np.random.permutation(len(dataloader.images))
    np.random.set_state(state)

    xs = dataloader.images[shuffle]
    ys = dataloader.labels[shuffle] if dataloader.labels is not None else None

    if not full:
        xs = xs[:1000]
        ys = ys[:1000] if ys is not None else None

    acc = 0.
    n = len(xs)
    bs = 200

    for i in xrange(0, n, bs):
        x = xs[i:i+bs] if dataloader.cast else xs[i:i+bs]
        y = ys[i:i+bs] if ys is not None else dataloader.labeler(x)
        acc += fn_acc(x, y) * len(x) / n

    summary = tf.Summary.Value(tag=tag, simple_value=acc)
    summary = tf.Summary(value=[summary])
    return acc, summary

# def approximate_accuracy(fn_acc, tag, dataloader, iters=200, bs=200):
#     acc = 0.

#     for i in xrange(iters):
#         x, y = dataloader.next_batch(bs)
#         acc += fn_acc(x, y) / iters

#     summary = tf.Summary.Value(tag=tag, simple_value=acc)
#     summary = tf.Summary(value=[summary])
#     return acc, summary

# def approximate_accuracy(M, tag, dataloader, iters=500, bs=200):
#     M.sess.run(M.test_acc_init)

#     for i in xrange(iters):
#         x, y = dataloader.next_batch(bs)
#         acc = M.sess.run(M.test_acc, {M.test_x: x, M.test_y: y})

#     summary = tf.Summary.Value(tag=tag, simple_value=acc)
#     summary = tf.Summary(value=[summary])
#     return acc, summary

def get_data(name, seed=0, npc=None, person=None):
    if name == 'svhn':
        return datasets.Svhn(seed=seed, npc=npc)

    elif name == 'digit':
        return datasets.SynDigits(seed=seed, npc=npc)

    elif name == 'mnist32':
        return datasets.Mnist(shape=(32, 32, 3), seed=seed, npc=npc)

    elif name == 'mnist28':
        return datasets.Mnist(shape=(28, 28, 3), seed=seed, npc=npc)

    elif name == 'mnistm32':
        return datasets.Mnistm(shape=(32, 32, 3), seed=seed, npc=npc)

    elif name == 'mnistm28':
        return datasets.Mnistm(shape=(28, 28, 3), seed=seed, npc=npc)

    elif name == 'gtsrb':
        return datasets.Gtsrb(seed=seed)

    elif name == 'sign':
        return datasets.SynSigns(seed=seed)

    elif name == 'cifar':
        return datasets.Cifar()

    elif name == 'stl':
        return datasets.Stl()

    elif name == 'pimnist':
        return datasets.Mnist(shape=(32, 32, 3), seed=seed, npc=npc, permute=True)

    elif name == 'pisvhn':
        return datasets.Svhn(seed=seed, npc=npc, permute=True)

    elif name == 'wifimexico':
        return datasets.WifiMexico(shape=(30, 400, 3), seed=seed, npc=npc, person=person)

    elif name == 'wifistanford':
        return datasets.WifiStanford(shape=(30, 400, 3), seed=seed, npc=npc)

    else:
        raise Exception('dataset {:s} not recognized'.format(name))
