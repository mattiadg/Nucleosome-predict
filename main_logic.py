from fasta_reader import get_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from conv_model import build_conv, build_lstm, build_conv_lstm
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import argparse
import os

def get_balanced_classes(idx, y):
    new_idx = np.zeros((idx.shape[0]), dtype=int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    p, n = 0, 0

    for i in range(len(new_idx)):
        if i % 2 == 0 and p < len(pos_idx):
            new_idx[-1-i] = pos_idx[p]
            p = p + 1
        elif i % 2 == 1 and n < len(neg_idx):
            new_idx[-1-i] = neg_idx[n]
            n = n + 1
        elif p >= len(pos_idx):
            new_idx[-1-i] = neg_idx[n]
            n = n + 1
        elif n >= len(neg_idx):
            new_idx[-1 - i] = pos_idx[p]
            p = p + 1

    return new_idx

def get_sequences(path):
    fin = open(path, 'r')
    buf = fin.readlines()
    fin.close()

    seqs, labels = [], []

    for i in range(len(buf)):
        label, seqRapp, out = buf[i].strip().split(";")

        ss = seqRapp.split(",")
        ss1 = list(map(float, ss))
        a = np.asarray(ss1)
        vv = np.reshape(a, (-1, 4), order='F')
        seqs.append(vv)
        labels.append(int(out))

    return np.array(seqs), np.array(labels)

def mix_indexes(samples_pos, samples_neg):
    samples_out = np.zeros((samples_pos.shape[0] + samples_neg.shape[0], samples_pos.shape[1], samples_pos.shape[2]))
    labels = np.zeros(samples_pos.shape[0] + samples_neg.shape[0])
    l, r, i = 0, 0, 0
    while l < len(samples_pos) and r < len(samples_neg):
        if i % 2 == 0:
            samples_out[i] = samples_pos[l]
            labels[i] = 1
            l += 1
        else:
            samples_out[i] = samples_neg[r]
            labels[i] = 0
            r += 1
        i += 1

    if l == len(samples_pos):
        samples_out[i:] = samples_neg[r:]
        labels[i:] = 1
    elif r == len(samples_neg):
        samples_out[i:] = samples_pos[l:]
        labels[i:] = 0

    return samples_out, labels

def split_train_test(labels, size):
    """
    :param data: Samples to split in train and test
    :param labels: labels of the data
    :param size: number of samples in the training set
    :return: (train_idx, test_idx)
    """
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    max_idx = int(size / 2)
    train_idx  = np.array(pos_idx[:max_idx].tolist() + neg_idx[:max_idx].tolist())
    test_idx  = np.array(pos_idx[max_idx:].tolist() + neg_idx[max_idx:].tolist())
    return train_idx, test_idx

def create_class_weights(y):
    pos_num, neg_num = len(y[y==1]), len(y[y==0])
    tot_num = len(y)
    pos_ratio, neg_ratio = pos_num / tot_num, neg_num / tot_num
    return {1: 1 - pos_ratio, 0: 1 - neg_ratio}

def dump_text(object_to_dump, outF):
    content = object_to_dump
    s = []
    for a, b in content:
        a = a.reshape(len(a)).tolist()
        b = b.reshape(len(b)).tolist()
        a = ';'.join([str(x) for x in a])
        b = ';'.join([str(x) for x in b])
        s.append(','.join([a, b]))
    outF.write('\n'.join(s))

labels2cat = {'Nucleosome' : [1, 0], 'Linker' : [0, 1]}
char2arr = {'A' : [1, 0, 0, 0], 'C' : [0, 1, 0, 0], 'G' : [0, 0, 1, 0], 'T' : [0, 0, 0, 1]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-species', required=True, type=str,
                        #choices=['Drosophila', 'Mouse', 'Yeast', 'Human'],
                        help='Select the dataset on which to perform the experiment')
    # Network parameters
    parser.add_argument('-filters', type=int, default=50,
                        help='Number of filters in the convolutional layer')
    parser.add_argument('-hidden_size', default=100, type=int,
                        help='Hidden size of the LSTM')
    parser.add_argument('-kernel_len', type=int, default=7,
                        help='Kernel length of the convolutional layer')

    # Training parameters
    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-k_fold', type=int, default=-1,
                        help='Number of folds for k-folds. -1 means random sampling')
    parser.add_argument('-folds', nargs='+',
                        help="""Choose a fraction of valid folds to compute""")
    parser.add_argument('-sample_test', action='store_true',
                        help="""If true, sample the test set and train only once""")
    parser.add_argument('-train_size', type=int, default=-1,
                        help="""If sample_test, select the size of the training set.""")
    parser.add_argument('-shuffle', type=bool, default=False,
                        help='Whether to shuffle the data or not')

    opt = parser.parse_args()

    #extract sequences and labels. Argument can be "sapiens", "melanogaster", "elegans"
    species_path = 'res/' + opt.species + '_rapp.csv'
    seqs, y = get_sequences(species_path)

    output = []

    if opt.sample_test:
        assert opt.train_size < len(seqs), "train_size must be smaller than the number of sequences!"
        if opt.train_size == -1:
            opt.train_size = len(seqs) / 2
        selector = lambda : train_test_split(seqs, y, train_size=opt.train_size)
        outfile = 'predictions_conv_lstm_sample100_' + opt.species + '.txt'
    else:
        selector = StratifiedKFold(n_splits=opt.k_fold, shuffle=opt.shuffle)
        fold_size = int(len(seqs) / opt.k_fold)
        outfile = 'predictions_conv_lstm_' + opt.species + '.txt'
    # For 100 times, it samples 100 test sentences and compute the results
    # for i in range(10):
    #     test_idx = np.random.choice(range(seqs.shape[0]), size=100, replace=True)
    #     train_idx = [x for x in range(seqs.shape[0]) if x not in test_idx]
    #     valid_idx = np.random.choice(train_idx, size=100, replace=False)
    #     train_idx = [x for x in train_idx if x not in valid_idx]
    #     X_train, X_valid, X_test = seqs[train_idx], seqs[valid_idx], seqs[test_idx]
    #     y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
    #     print(X_train.shape, y_train.shape)
        #Build new network
        #model = build_lstm(4)
    i = 0

    if not os.path.exists(opt.species):
        os.makedirs(opt.species)

    if opt.sample_test:
        X_train, X_test_pool, y_train, y_test_pool = selector()
        #X_test_pool, y_test_pool = seqs[test_idx], y[test_idx]
        #seqs, y = seqs[train_idx], y[train_idx]
        #seqs, labels = mix_indexes(seqs[np.where(y == 1)], seqs[np.where(y == 0)])
        max_valid = min([1000, int(len(seqs)*0.1)])
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=max_valid)
        #y_train, y_valid = y_train[max_valid:], y_train[:max_valid]
        
        ################################################################
        model = build_conv_lstm(opt.filters, opt.kernel_len, opt.hidden_size)
        ################################################################
        
        early = EarlyStopping(patience=3)
        mcp = ModelCheckpoint(opt.species + '/iter' + str(i) + "." + opt.species +".model", save_best_only=True)
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=64,
                  callbacks=[early, mcp], epochs=opt.epochs, class_weight=create_class_weights(y_train))
        for i in range(100):
            test_idx = np.random.choice(X_test_pool.shape[0], size=100, replace=True)
            X_test, y_test = X_test_pool[test_idx], y_test_pool[test_idx]
            pred = model.predict(X_test, batch_size=64)
            ord_sort = np.argsort(pred, axis=None)
            output += [(pred[ord_sort], y_test[ord_sort])]

    else:
        valid_folds = list(range(int(opt.folds[0]), int(opt.folds[1]))) if opt.folds else list(range(opt.k_fold))

        for train_idx, test_idx in selector.split(seqs, y):
            if i in valid_folds:
                X_train, X_test = seqs[train_idx], seqs[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                max_valid = min([1000, int(len(seqs)*0.1)])
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=max_valid)
                model = build_conv_lstm(opt.filters, X_train.shape[1], opt.kernel_len, opt.hidden_size)
                early = EarlyStopping(patience=9)
                mcp = ModelCheckpoint(opt.species + '/iter' + str(i) + "." + opt.species +".model", save_best_only=True)
                model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=64,
                          callbacks=[early, mcp], epochs=opt.epochs, class_weight=create_class_weights(y_train))
                pred = model.predict(X_test, batch_size=64)
                output += [(pred, y_test)]
            i += 1

    with open(outfile, 'w') as f:
        dump_text(output, f)

    print("This is output:", output)
    print("Finished")
