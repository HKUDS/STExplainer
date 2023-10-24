from email import utils
import torch
import time
from engine import trainer, record_metric
import numpy as np
import utils.util as util
from data.DataHandler_st import DataHandler
from Params import args, logger
import os
import random
# import tensorflow as tf

seed = 777

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # tf.random.set_seed(seed)
    torch.backends.cudnn.enabled = True

def train():
    set_seed(seed)
    
    handler = DataHandler()
    tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w, temp_adj= handler.get_dataloader()
    
    engine = trainer(scaler, sp_adj, sp_adj_w, temp_adj)
    tra_val_metric = dict()
    if args.testonly is not True:
        logger.info('start training .....')
        for epoch in range(1, args.max_epoch+1):
            print('*'*20, 'Training Process', '*'*20)
            t1 = time.time()
            tra_val_metric= engine.train(epoch, tra_loader, tra_val_metric)
            t2 = time.time()
            print('*'*20, 'Validating Process', '*'*20)
            tra_val_metric, stopFlg = engine.validation(epoch, val_loader, tra_val_metric)
            tra_val_metric = record_metric(tra_val_metric, [t2 - t1], ['cost time'])
            logger.info(tra_val_metric)
            if stopFlg:
                break
        logger.info('start testing .....')
        engine.test(tst_loader)        

    else:
        logger.info('start testing .....')
        engine.test(tst_loader)    


if __name__ == "__main__":
    t1 = time.time()
    train()
    t2 = time.time()
    logger.info("Total time spent: {:.4f}".format(t2 - t1))

