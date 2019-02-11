# coding: utf-8
import sys
sys.path.append('..')
import numpy
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from common.np import *  # import numpy as np
from common.util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
        self.err_train = []
        self.err_test = []

    def fit(self, x_train, t_train, x_test, t_test, max_epoch=10, batch_size=32, max_grad=None, eval_interval=10):
        data_size = len(x_train)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            epoch_start_time = time.time()
            # シャッフル
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x_train[idx]
            t = t_train[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 評価
                # if (eval_interval is not None) and (iters % eval_interval) == 0:
                #     avg_loss = total_loss / loss_count
                #     elapsed_time = time.time() - start_time
                #     print('| epoch %d |  iter %d / %d | time %d[s] | loss %.5f'
                #           % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                #     # self.loss_list.append(float(avg_loss))
                #     total_loss, loss_count = 0, 0


            def get_error_rate(xs, ts):
                data_size = xs.shape[0]
                max_iter, mod = divmod(data_size, batch_size)
                acc_count = 0  # 正答数
                for iters in range(max_iter):
                    batch_x = xs[iters*batch_size:(iters+1)*batch_size]
                    batch_t =ts[iters*batch_size:(iters+1)*batch_size]
                    acc_count += self.model.validation(batch_x, batch_t)
                # データサイズがバッチサイズで割りきれない場合
                if mod > 0:
                    batch_x = xs[max_iter*batch_size:]
                    batch_t = ts[max_iter*batch_size:]
                    acc_count += self.model.validation(batch_x, batch_t)

                return 1 - acc_count / data_size

            # loss, accuracyの算出と表示
            display_epoch = self.current_epoch + 1
            self.loss_list.append(total_loss / loss_count)
            total_loss, loss_count = 0, 0
            elapsed_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            print('| epoch %d | time/total %d/%d[s] | loss = %.5f'
                  % (display_epoch, elapsed_time, total_time, self.loss_list[-1]))

            # errorの算出は5epochまではepoch毎、以降はeval_interval毎に行う
            if (display_epoch <= 5) or (display_epoch == max_epoch) or (eval_interval is not None) and (display_epoch % eval_interval) == 0:
                self.err_train.append(get_error_rate(x_train, t_train))
                elapsed_time = time.time() - epoch_start_time
                total_time = time.time() - start_time
                print('| epoch %d | time/total %d/%d[s] | train error = %.5f'
                      % (display_epoch, elapsed_time, total_time, self.err_train[-1]))
                self.err_test.append(get_error_rate(x_test, t_test))
                elapsed_time = time.time() - epoch_start_time
                total_time = time.time() - start_time
                print('| epoch %d | time/total %d/%d[s] | test error  = %.5f'
                      % (display_epoch, elapsed_time, total_time, self.err_test[-1]))

            self.current_epoch += 1

    def plot(self, image_path=None, log_scale=False):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # 色の設定
        color_loss = 'blue'
        color_err = 'orange'
        epoch_range = range(1, len(self.loss_list)+1)
        x1 = np.array(epoch_range)
        x2 = np.array([x for x in epoch_range if x <= 5 or x == len(self.loss_list) or x % self.eval_interval == 0])
        ax1.plot(x1, self.loss_list, color=color_loss, label='loss')
        ax2.plot(x2, self.err_train, color=color_err, label='train error')
        ax2.plot(x2, self.err_test, color=color_err, linestyle='dashed', label='test error')
        ax1.tick_params(axis='y', colors=color_loss)
        ax2.tick_params(axis='y', colors=color_err)
        # 軸ラベル
        plt.xlabel('epoch')
        plt.xlim((1, len(self.loss_list)+1))
        if log_scale:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
        else:
            ax2.set_ylim((0.0, 1.0))
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Error')
        ax1.set_title("train error={:.5f}  test error={:.5f}".format(self.err_train[-1], self.err_test[-1]))
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right')
        ax2.legend(bbox_to_anchor=(1, 0.9), loc='upper right')
        #グリッド表示(ax2のみ)
        ax2.grid(True)
        if image_path is not None:
            plt.savefig(image_path)
        plt.show(block=False)

    def clear(self):
        self.current_epoch = 0
        self.loss_list = []
        self.err_train = []
        self.err_test = []


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # パープレキシティの評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()


def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
