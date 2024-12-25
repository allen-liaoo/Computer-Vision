import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def get_mini_batch(im_train, label_train, batch_size):
    size = im_train.shape[1]
    # shuffle images and labels
    idxs = np.random.randint(low=0, high=im_train.shape[1], size=size)
    im_train = im_train[:, idxs]
    label_train = label_train[:, idxs]

    num_batches = size // batch_size
    batches_end = num_batches * batch_size + 1

    # split into batches, then append leftover elements as the last batch
    mini_batch_x = np.split(im_train[:batches_end], num_batches, axis=1)
    if len(im_train[batches_end:]) != 0:
        mini_batch_x.extend(im_train[batches_end:])
    mini_batch_y = np.split(label_train[:batches_end], num_batches, axis=1)
    if len(label_train[batches_end:]) != 0:
        mini_batch_y.extend(label_train[batches_end:])

    # convert mini_batch_y to one-hot encoded vectors
    for i, batch in enumerate(mini_batch_y):
        mini_batch_y[i] = np.zeros((10, batch_size))
        mini_batch_y[i][batch, np.arange(batch_size)] = 1

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = w @ x + b
    return y


def fc_backward(dl_dy, x, w, b):
    dl_dx = np.sum(w * dl_dy, axis=0)[:,None] # first number of dl_dy is multiplied with first column of w via broadcasting
    dl_dw = dl_dy @ x.T # outer product, w_ij' = y_i' * x_j
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def relu(x):
    zeroes = np.zeros(x.shape)
    y = np.maximum(x, zeroes)
    return y


def relu_backward(dl_dy, x):
    dl_dx = np.where(x < 0, 0, 1)
    return dl_dy * dl_dx


def loss_cross_entropy_softmax(x, y):
    # consulted https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    y_curly = np.exp(x) # e^x
    y_curly = y_curly / np.sum(y_curly)
    l = -np.sum(y * np.log(y_curly))

    dl_dx = y_curly - y
    return l, dl_dx


def conv(x, w_conv, b_conv):
    # kernel size: 3 x 3, stride 1
    # pad x with 0s on the edges
    kernel_size = 3
    pad_size = kernel_size // 2
    x = np.pad(x, ((pad_size,pad_size),(pad_size,pad_size),(0,0)), constant_values=-1)

    # convolution
    x_windows = np.lib.stride_tricks.sliding_window_view(x[:,:,0], (kernel_size, kernel_size))
    x_windows = x_windows[..., None, None] # restore input channel (of 1) and add new axis for broadcasting to output channel values
    conv_t = w_conv * x_windows
    conv_t = np.sum(conv_t, axis=(2,3,4)) # sum over (h,w,C1) of (H,W,h,w,C1,C2), gets (H,W,C2)
    y = conv_t + b_conv[:,0]
    return y


def conv_backward(dl_dy, x, w_conv, b_conv):
    kernel_size = 3
    H, W, _ = x.shape
    # for each element in w that is k x k, its derivative concerns the whole of x except for k-1 rows and columns
    # so we create another sliding window across x with width and height as W-k+1 and H-k+1
    x_windows = np.lib.stride_tricks.sliding_window_view(x, (H-kernel_size+1, W-kernel_size+1), axis=(0,1))
    y_windows = np.lib.stride_tricks.sliding_window_view(dl_dy, (H-kernel_size+1, W-kernel_size+1), axis=(0,1))

    # derivative of w_ij across channels uses the same windows of x as scalar, but different parts of dl_dy
    dl_dw = x_windows * y_windows # element-wise multiply
    dl_dw = np.sum(dl_dw, axis=(3,4)) # sum over all windows
    dl_dw = dl_dw[:,:,None,:] # (h,w,C2) to (h,w,C1,C2) (C1 = 1, so this is hardcoded. What is supposed to happened to it?)
    dl_db = np.sum(dl_dy, axis=(0,1))[:,None] # derivative is sum of dl_dy * 1; sum over first two axis of (H,W,C2)
    return dl_dw, dl_db


# return ndarray of pooling windows
def pool2x2_windows(x):
    pool_size = 2
    stride = pool_size
    H, W, _ = x.shape

    # create pooling windows, which has stride 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (pool_size, pool_size), axis=(0,1))

    # skip over windows within stride
    h_idxs_stride = np.arange(0, H-pool_size+1, stride)
    w_idxs_stride = np.arange(0, W-pool_size+1, stride)
    x_idxs_stride, y_idxs_stride = np.meshgrid(h_idxs_stride, w_idxs_stride)
    windows = windows[x_idxs_stride, y_idxs_stride, ...]
    return windows


def pool2x2(x):
    windows = pool2x2_windows(x)
    y = np.max(windows, axis=(3,4))
    return y


def pool2x2_backward(dl_dy, x):
    pool_size = 2
    windows = pool2x2_windows(x)

    windows = np.reshape(windows, windows.shape[:-2] + (-1,)) # flatten each window (for argmax)
    maxes = np.argmax(windows, axis=3)
    dl_dx_windows = np.zeros_like(windows)

    # build indices for dl_dx to set 1 to maxes
    max_indexes = (np.arange(windows.shape[0])[:, None, None], # H
                    np.arange(windows.shape[1])[None, :, None], # W
                    np.arange(windows.shape[2])[None, None, :], # Channel
                    maxes) # max value index per window
    dl_dx_windows[max_indexes] = 1

    # multiply with dl_dy
    dl_dx_windows = dl_dy[..., None] * dl_dx_windows

    # reverse window back to image shape by
    # assigning derivative matrix (1s at max of each pooling window, 0s elsewhere) to dl_dx of (H,W,...)
    # each assignment writes a window to dl_dx
    dl_dx = np.zeros_like(x)
    H_wins, W_wins, *_ = dl_dx_windows.shape

    dl_dx_windows = np.reshape(dl_dx_windows, dl_dx_windows.shape[:-1] + (pool_size, pool_size)) # unflatten window
    for i in range(H_wins):
        row = i * pool_size
        for j in range(W_wins):
            col = j * pool_size
            dl_dx[row : row + pool_size,
                  col : col + pool_size, ...] = np.transpose(dl_dx_windows[i, j, ...], axes=(1,2,0)) # (C, p, p) -> (p, p, C), where p is pool_size
    return dl_dx


def flattening(x):
    return x.reshape((-1, 1))


def flattening_backward(dl_dy, x):
    return dl_dy.reshape(x.shape)


def train_mlp(mini_batch_x, mini_batch_y, learning_rate=0.1, decay_rate=0.95, num_iters=10000):
    nh = 30

    # initialization network weights
    w1 = np.random.randn(nh, 196)  # (30, 196)
    b1 = np.zeros((nh, 1))  # (30, 1)
    w2 = np.random.randn(10, nh)  # (10, 30)
    b2 = np.zeros((10, 1))  # (10, 1)

    k = 0
    losses = np.zeros((num_iters, 1))
    for iter in tqdm(range(num_iters), desc='Training MLP'):
        if (iter + 1) % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dl_dw1_batch = np.zeros((nh, 196))
        dl_db1_batch = np.zeros((nh, 1))
        dl_dw2_batch = np.zeros((10, nh))
        dl_db2_batch = np.zeros((10, 1))
        batch_size = mini_batch_x[k].shape[1]
        ll = np.zeros((batch_size, 1))
        for i in range(batch_size):
            x = mini_batch_x[k][:, [i]]
            y = mini_batch_y[k][:, [i]]

            # forward propagation
            h1 = fc(x, w1, b1)
            h2 = relu(h1)
            h3 = fc(h2, w2, b2)

            # loss computation (forward + backward)
            l, dl_dy = loss_cross_entropy_softmax(h3, y)
            ll[i] = l

            # backward propagation
            dl_dh2, dl_dw2, dl_db2 = fc_backward(dl_dy, h2, w2, b2)
            dl_dh1 = relu_backward(dl_dh2, h1)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dh1, x, w1, b1)

            # accumulate gradients
            dl_dw1_batch += dl_dw1
            dl_db1_batch += dl_db1
            dl_dw2_batch += dl_dw2
            dl_db2_batch += dl_db2

        losses[iter] = np.mean(ll)
        k = k + 1
        if k > len(mini_batch_x) - 1:
            k = 0

        # accumulate gradients
        w1 -= learning_rate * dl_dw1_batch / batch_size
        b1 -= learning_rate * dl_db1_batch / batch_size
        w2 -= learning_rate * dl_dw2_batch / batch_size
        b2 -= learning_rate * dl_db2_batch / batch_size

    return w1, b1, w2, b2, losses


def train_cnn(mini_batch_x, mini_batch_y, learning_rate=0.05, decay_rate=0.95, num_iters=10000):
    # initialization network weights
    w_conv = 0.1 * np.random.randn(3, 3, 1, 3)
    b_conv = np.zeros((3, 1))
    w_fc = 0.1 * np.random.randn(10, 147)
    b_fc = np.zeros((10, 1))

    k = 0
    losses = np.zeros((num_iters, 1))
    for iter in tqdm(range(num_iters), desc='Training CNN'):
        if (iter + 1) % 1000 == 0:
            learning_rate = decay_rate * learning_rate
            # print('iter {}/{}'.format(iter + 1, num_iters))

        dl_dw_conv_batch = np.zeros(w_conv.shape)
        dl_db_conv_batch = np.zeros(b_conv.shape)
        dl_dw_fc_batch = np.zeros(w_fc.shape)
        dl_db_fc_batch = np.zeros(b_fc.shape)
        batch_size = mini_batch_x[k].shape[1]
        ll = np.zeros((batch_size, 1))

        for i in range(batch_size):
            x = mini_batch_x[k][:, [i]].reshape((14, 14, 1))
            y = mini_batch_y[k][:, [i]]

            # forward propagation
            h1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
            h2 = relu(h1)  # (14, 14, 3)
            h3 = pool2x2(h2)  # (7, 7, 3)
            h4 = flattening(h3)  # (147, 1)
            h5 = fc(h4, w_fc, b_fc)  # (10, 1)

            # loss computation (forward + backward)
            l, dl_dy = loss_cross_entropy_softmax(h5, y)
            ll[i] = l

            # backward propagation
            dl_dh4, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, h4, w_fc, b_fc)  # (147, 1), (10, 147), (10, 1)
            dl_dh3 = flattening_backward(dl_dh4, h3)  # (7, 7, 3)
            dl_dh2 = pool2x2_backward(dl_dh3, h2)  # (14, 14, 3)
            dl_dh1 = relu_backward(dl_dh2, h1)  # (14, 14, 3)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dh1, x, w_conv, b_conv)  # (3, 3, 1, 3), (3, 1)

            # accumulate gradients
            dl_dw_conv_batch += dl_dw_conv
            dl_db_conv_batch += dl_db_conv
            dl_dw_fc_batch += dl_dw_fc
            dl_db_fc_batch += dl_db_fc

        losses[iter] = np.mean(ll)
        k = k + 1
        if k > len(mini_batch_x) - 1:
            k = 0

        # update network weights
        w_conv -= learning_rate * dl_dw_conv_batch / batch_size
        b_conv -= learning_rate * dl_db_conv_batch / batch_size
        w_fc -= learning_rate * dl_dw_fc_batch / batch_size
        b_fc -= learning_rate * dl_db_fc_batch / batch_size

    return w_conv, b_conv, w_fc, b_fc, losses


def visualize_training_progress(losses, num_batches):
    # losses - (n_iter, 1)
    num_iters = losses.shape[0]
    num_epochs = math.ceil(num_iters / num_batches)
    losses_epoch = np.zeros((num_epochs, 1))
    losses_epoch[:num_epochs-1, 0] = np.mean(
        np.reshape(losses[:(num_epochs - 1)*num_batches], (num_epochs - 1, num_batches)), axis=1)
    losses_epoch[num_epochs-1] = np.mean(losses[(num_epochs - 1)*num_batches:])

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].plot(range(num_iters), losses), axs[0].set_title('Training loss w.r.t. iteration')
    axs[0].set_xlabel('Iteration'), axs[0].set_ylabel('Loss'), axs[0].set_ylim([0, 5])
    axs[1].plot(range(num_epochs), losses_epoch), axs[1].set_title('Training loss w.r.t. epoch')
    axs[1].set_xlabel('Epoch'), axs[1].set_ylabel('Loss'), axs[1].set_ylim([0, 5])
    fig.suptitle('MLP Training Loss', fontsize=16)
    plt.show()


def infer_mlp(x, w1, b1, w2, b2):
    # x - (m, 1)
    h1 = fc(x, w1, b1)
    h2 = relu(h1)
    h3 = fc(h2, w2, b2)
    y = np.argmax(h3)
    return y


def infer_cnn(x, w_conv, b_conv, w_fc, b_fc):
    # x - (H(14), W(14), C_in(1))
    h1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
    h2 = relu(h1)  # (14, 14, 3)
    h3 = pool2x2(h2)  # (7, 7, 3)
    h4 = flattening(h3)  # (147, 1)
    h5 = fc(h4, w_fc, b_fc)  # (10, 1)
    y = np.argmax(h5)
    return y


def compute_confusion_matrix_and_accuracy(pred, label, n_classes):
    # pred, label - (n, 1)
    accuracy = np.sum(pred == label) / len(label)
    confusion = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        for i in range(n_classes):
            # ground true is j but predicted to be i
            confusion[i, j] = np.sum(np.logical_and(label == j, pred == i)) / label.shape[0]
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("Accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


def get_MNIST_data(resource_dir):
    with open(resource_dir + 'mnist_train.npz', 'rb') as f:
        d = np.load(f)
        image_train, label_train = d['img'], d['label']  # (12k, 14, 14), (12k, 1)
    with open(resource_dir + 'mnist_test.npz', 'rb') as f:
        d = np.load(f)
        image_test, label_test = d['img'], d['label']  # (2k, 14, 14), (2k, 1)

    label_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return image_train, label_train, image_test, label_test, label_classes


if __name__ == '__main__':
    work_dir = 'hw5/'

    image_train, label_train, image_test, label_test, label_classes = get_MNIST_data(work_dir)
    image_train, image_test = image_train.reshape((-1, 196)).T / 255.0, image_test.reshape((-1, 196)).T / 255.0

    # Part 1: Multi-layer Perceptron
    # train
    # mini_batch_x, mini_batch_y = get_mini_batch(image_train, label_train.T, batch_size=32)
    # w1, b1, w2, b2, losses = train_mlp(mini_batch_x, mini_batch_y,
    #                                    learning_rate=0.1, decay_rate=0.9, num_iters=10000)
    # visualize_training_progress(losses, len(mini_batch_x))
    # np.savez(work_dir + 'mlp.npz', w1=w1, b1=b1, w2=w2, b2=b2)

    # test
    # pred_test = np.zeros_like(label_test)
    # for i in range(image_test.shape[1]):
    #     pred_test[i, 0] = infer_mlp(image_test[:, i].reshape((-1, 1)), w1, b1, w2, b2)
    # confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    # visualize_confusion_matrix(confusion, accuracy, label_classes)

    # Part 2: Convolutional Neural Network
    # train
    np.random.seed(0)
    mini_batch_x, mini_batch_y = get_mini_batch(image_train, label_train.T, batch_size=32)
    w_conv, b_conv, w_fc, b_fc, losses = train_cnn(mini_batch_x, mini_batch_y,
                                                   learning_rate=0.1, decay_rate=0.9, num_iters=1000)
    visualize_training_progress(losses, len(mini_batch_x))
    np.savez(work_dir + 'cnn.npz', w_conv=w_conv, b_conv=b_conv, w_fc=w_fc, b_fc=b_fc)

    # test
    pred_test = np.zeros_like(label_test)
    for i in range(image_test.shape[1]):
        pred_test[i, 0] = infer_cnn(image_test[:, i].reshape((14, 14, 1)), w_conv, b_conv, w_fc, b_fc)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)

