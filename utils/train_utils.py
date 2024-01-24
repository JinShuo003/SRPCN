import json
import os
import torch

import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

from utils.log_utils import LogFactory


def get_dataloader(dataset_class, specs: dict):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    data_source = specs.get("DataSource")
    train_split_file = specs.get("TrainSplit")
    test_split_file = specs.get("TestSplit")
    trian_options = specs.get("TrainOptions")
    batch_size = trian_options.get("BatchSize")
    num_data_loader_threads = trian_options.get("DataLoaderThreads")

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    train_dataset = dataset_class(data_source, train_split)
    test_dataset = dataset_class(data_source, test_split)
    logger.info("length of train_dataset: {}".format(train_dataset.__len__()))
    logger.info("length of test_dataset: {}".format(test_dataset.__len__()))

    # get dataloader
    train_dataloader = data_utils.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logger.info("length of train_dataloader: {}".format(train_dataloader.__len__()))
    logger.info("length of test_dataloader: {}".format(test_dataloader.__len__()))

    return train_dataloader, test_dataloader


def get_checkpoint(specs):
    device = specs.get("Device")
    pre_train = specs.get("TrainOptions").get("PreTrain")
    continue_train = specs.get("TrainOptions").get("ContinueTrain")
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    assert not (pre_train and continue_train)

    checkpoint = None
    if pre_train:
        logger.info("pretrain mode")
        pretrain_model_path = specs.get("TrainOptions").get("PreTrainModel")
        logger.info("load checkpoint from {}".format(pretrain_model_path))
        checkpoint = torch.load(pretrain_model_path, map_location="cuda:{}".format(device))
    elif continue_train:
        logger.info("continue train mode")
        continue_from_epoch = specs.get("TrainOptions").get("ContinueFromEpoch")
        para_save_dir = specs.get("ParaSaveDir")
        para_save_path = os.path.join(para_save_dir, specs.get("TAG"))
        checkpoint_path = os.path.join(para_save_path, "epoch_{}.pth".format(continue_from_epoch))
        logger.info("load checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cuda:{}".format(device))
    return checkpoint


def get_network(specs, model_class, checkpoint, *args):
    device = specs.get("Device")
    logger = LogFactory.get_logger(specs.get("LogOptions"))

    network = model_class(*args).to(device)

    if checkpoint:
        logger.info("load model parameter from epoch {}".format(checkpoint["epoch"]))
        network.load_state_dict(checkpoint["model"])
    
    return network


def get_optimizer(specs, network, checkpoint):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    init_lr = specs.get("TrainOptions").get("LearningRateOptions").get("InitLearningRate")
    step_size = specs.get("TrainOptions").get("LearningRateOptions").get("StepSize")
    gamma = specs.get("TrainOptions").get("LearningRateOptions").get("Gamma")
    logger.info("init_lr: {}, step_size: {}, gamma: {}".format(init_lr, step_size, gamma))

    pre_train = specs.get("TrainOptions").get("PreTrain")
    continue_train = specs.get("TrainOptions").get("ContinueTrain")
    assert not (pre_train and continue_train)
    
    if continue_train:
        last_epoch = specs.get("TrainOptions").get("ContinueFromEpoch")
        optimizer = torch.optim.Adam([{'params': network.parameters(), 'initial_lr': init_lr}], lr=init_lr, betas=(0.9, 0.999))
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

        logger.info("load lr_schedule parameter from epoch {}".format(checkpoint["epoch"]))
        lr_schedule.load_state_dict(checkpoint["lr_schedule"])
        logger.info("load optimizer parameter from epoch {}".format(checkpoint["epoch"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=init_lr, betas=(0.9, 0.999))
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return lr_schedule, optimizer


def get_tensorboard_writer(specs):
    writer_path = os.path.join(specs.get("TensorboardLogDir"), specs.get("TAG"))
    if not os.path.isdir(writer_path):
        os.makedirs(writer_path)

    return SummaryWriter(writer_path)


def save_model(specs, model, lr_schedule, optimizer, epoch):
    para_save_dir = specs.get("ParaSaveDir")
    para_save_path = os.path.join(para_save_dir, specs.get("TAG"))
    if not os.path.isdir(para_save_path):
        os.mkdir(para_save_path)
    
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    checkpoint_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))

    torch.save(checkpoint, checkpoint_filename)


def get_loss_weight(loss_options, epoch):
    begin_epoch = loss_options.get("BeginEpoch")
    init_ratio = loss_options.get("InitRatio")
    step_size = loss_options.get("StepSize")
    gamma = loss_options.get("Gamma")
    if epoch < begin_epoch:
        return 0
    return init_ratio * pow(gamma, int((epoch - begin_epoch) / step_size))


def record_loss_info(specs: dict, tag: str, avrg_loss, epoch: int, tensorboard_writer: SummaryWriter):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    tensorboard_writer.add_scalar("{}".format(tag), avrg_loss, epoch)
    logger.info('{}: {}'.format(tag, avrg_loss))

