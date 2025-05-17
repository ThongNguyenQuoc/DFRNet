from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.dataset = "ms1mv2"
config.loss = "adaface"
config.network = "r50"
config.resume = True
config.output = "/kaggle/working/output"
config.embedding_size = 512
config.sample_rate = 0.1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1  # batch size is 512


config.rec =  "/kaggle/input/ms1mv2/faces_emore/faces_emore"
config.num_classes =  85742
config.num_image = 5822653
config.num_epoch = 40
config.warmup_epoch = -1
config.decay_epoch = [8, 12, 15, 18]
config.val_targets = ["lfw"]