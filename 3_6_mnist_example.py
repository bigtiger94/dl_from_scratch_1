import time
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import sys, os
import pickle
import gzip
import numpy as np
sys.path.append(os.pardir)
# from dataset.mnist import load_mnist

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    if os.path.exists(file_path):
        return
    
    print("Downloading " + file_name + "...")
    time.sleep(5)
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

def download_mnist():
    for v in key_file.values():
        _download(v)

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print("Converting " + file_name + " to np array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset = 8)
    print("Done")
    return(labels)

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    print("Converting " + file_name + " to np array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset = 16)
    data = data.reshape(-1, img_size)
    print("Done")
    return(data)

def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    return(dataset)

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pkl file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done")
    
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_mnist(normalize = True, flatten = True, one_hot_label = False):
    if not os.path.exists(save_file):
        init_mnist()
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
        
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    return((dataset['train_img'], dataset['train_label']),
           (dataset['test_img'], dataset['test_label']))
           
if __name__ == '__main__':
    
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"mnist_data")
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    save_file = dataset_dir + "/mnist.pkl"

    train_num = 60000
    test_num = 10000
    img_dim = (1, 28, 28)
    img_size = 784
    init_mnist()
    
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize = False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)

    
def softmax(val):
    C = np.max(val)
    exp_val = np.exp(val-C)
    y = exp_val / np.sum(exp_val)
    return(y)


# (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

