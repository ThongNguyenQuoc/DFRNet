from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "adaface"
config.network = "r50"
config.resume = False
config.output = "/kaggle/working/work_dirs"
config.embedding_size = 512
config.sample_rate = 0.1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1  # batch size is 512

config.rec = "/kaggle/input/casia-webface/faces_webface_112x112"
config.num_classes = 10575
config.num_image = 494414
config.num_epoch = 15
config.warmup_epoch = -1
config.decay_epoch = [2, 7, 12]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.dataset = "casia-webface"
