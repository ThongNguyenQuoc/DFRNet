import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter


class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.current_highest = 0
        if self.rank == 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, module_partial_fc, backbone: torch.nn.Module, global_step: int):
        results = []
        current_score = 0
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info('[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)
            current_score += acc2
        logging.info('[+][%d]Score / Score-Highest: %2.5f / %2.5f' % (global_step, current_score, self.current_highest))
        if self.current_highest <= current_score:
            # Sửa ở đây: dùng thư mục /kaggle/working/tmp thay vì 'tmp'
            tmp_dir = '/kaggle/working/tmp'
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir, exist_ok=True)
            path_save = os.path.join(tmp_dir, 'backbone_{}.pth'.format(current_score))
            torch.save(backbone.module.state_dict(), path_save)
            print('Saved as best checkpoint to', path_save)
            # Nếu cần lưu các tham số khác của module_partial_fc, bạn có thể mở phần comment bên dưới
            # if global_step > 100 and module_partial_fc is not None:
            #     module_partial_fc.save_params(folder="tmp")
            self.current_highest = current_score


    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, module_partial_fc, backbone: torch.nn.Module):
        if self.rank == 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(module_partial_fc, backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, global_step, backbone, partial_fc, frequent = None):
        if frequent is not None:
            if global_step > 100 and self.rank == 0 and global_step % frequent == 0:
                path_module = os.path.join(self.output, "backbone.pth")
                torch.save(backbone.module.state_dict(), path_module)
                logging.info("Pytorch Model Saved in '{}'".format(path_module))

            if global_step > 100 and partial_fc is not None and global_step % frequent == 0:
                partial_fc.save_params()
        else:
            if global_step > 100 and self.rank == 0:
                path_module = os.path.join(self.output, "backbone.pth")
                torch.save(backbone.module.state_dict(), path_module)
                logging.info("Pytorch Model Saved in '{}'".format(path_module))

            if global_step > 100 and partial_fc is not None:
                partial_fc.save_params()
