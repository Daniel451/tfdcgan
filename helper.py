import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

import numpy as np

from itertools import product
from string import ascii_lowercase
import datetime
import random



class Helper(object):


    @staticmethod
    def print_variables(tfvars, name=""):
        print("\n" + name)
        for v in tfvars:
            print(v)


    @staticmethod
    def PAI(arr : np.ndarray, name=""):
        """
        print array information
        """
        shapestr = str(arr.shape).rjust(20, " ")
        dtypestr = str(arr.dtype).rjust(10, " ")

        pstr = ""
        if name != "":
            pstr = "[{}] ".format(name).ljust(20, " ")
        pstr += "shape: {} | dtype: {} | min: {:>10.5f} | max: {:>10.5f} | mean: {:>10.5f}" \
            .format(shapestr, dtypestr, np.min(arr), np.max(arr), np.mean(arr))

        print(pstr)


    @staticmethod
    def print_costs(costs : float, name : str, epoch : int, run : int):
        """
        :param costs: current costs
        :param name: label of the costs
        :param epoch: current epoch
        :param run: current run / training iteration
        :return:
        """
        print("[epoch {:>5} | run {:>5}] {:>20} : {:>10.4f}".format(epoch, run, name, costs))


    @staticmethod
    def plot_batch_to_disk(img_batch : np.ndarray, fpath : str, fname : str, plot_title : str):
        batch_size = img_batch.shape[0]

        img_container = np.zeros((352, 352)).astype(np.float32)
        img_container[1, :] = 1.0
        img_container[:, 1] = 1.0
        img_container[351, :] = 1.0
        img_container[:, 351] = 1.0

        for index, (row, col) in zip(range(10*10), product(range(10), range(10))):
            if index >= batch_size:
                break
            pos_row = 5 + row * 35
            pos_col = 5 + col * 35
            single_image_data = img_batch[index].reshape((28, 28))

            img_container[pos_row:pos_row+28, pos_col:pos_col+28] = single_image_data

        # img_name = "img_epoch_{:0>3}_run_{:0>4}.jpg".format(e, run)
        # img_path = args.img_save_path

        f, ax = plt.subplots()
        datetime_str = datetime.datetime.now().strftime("[%Y-%m-%d | %H:%M]")
        ax.set_title("{} {}".format(datetime_str, plot_title))
        cax = ax.imshow(img_container, cmap="gray", interpolation="none")
        f.colorbar(cax)
        f.savefig(fpath + fname)
        plt.close(f)


    @staticmethod
    def get_random_str(length):
        chars = ascii_lowercase + "0123456789"
        return "".join(random.choices(chars, k=length))
