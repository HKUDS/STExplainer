import argparse
from utils.util import get_logger
import os
import yaml
import datetime
def read_name(config_path):
    nameList = config_path.split('/')
    allName = nameList[-1]
    nameList = allName.split('.')
    return nameList[0]
def genPath(path_str, date):
    return path_str+date+'/'
def print_args(args):
    log_level = 'INFO'
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(args.logdir, __name__, 'info_{}.log'.format(args.name), level=log_level)
    logger.info(args.config)
    # logger.info(args.dist_thr)

    return logger
def arg4config():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--config', default='./config/BikeNYC/BikeNYC.yaml', type=str)

    args = parser.parse_args()
    config_path = args.config
    return config_path



class ParamConfig:
    # def __init__(self):
    config_path = arg4config()

    print('config: {}'.format(config_path))
    with open(config_path, 'r', encoding='utf-8') as y:
        config = yaml.safe_load(y)
    # control:
    model = config['control']['model']
    testonly = True if config['control']['testonly'] else False
    device = config['control']['device']
    date = config['control']['date']
    name = read_name(config_path)
    mdir = genPath(config['control']['mdir'], date)
    logdir = genPath(config['control']['logdir'], date)

    # data:
    dataset = config['data']['dataset']
    data_path = config['data']['data_path']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    lag = config['data']['lag']
    horizon = config['data']['horizon']  
    batch_size = config['data']['batch_size'] 
    geo_graph = config['data']['geo_graph']  
    num_nodes = config['data']['num_nodes'] 
    adj_filename = config['data']['adj_filename']
    id_filename = config['data']['id_filename']

    # training:
    patience = config['training']['patience']
    model_path = config['training']['model_path']
    mae_thresh = config['training']['mae_thresh']
    mape_thresh = config['training']['mape_thresh']
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    lr_decay_ratio = config['training']['lr_decay_ratio']
    steps = config['training']['steps']
    criterion = config['training']['criterion']
    max_epoch = config['training']['max_epoch']
    grad_norm = True if config['training']['grad_norm'] else False  

    # 
    popu_norm = True
    mask_neg = True  
    scaler_norm = True
    start_ori = datetime.datetime.strptime('2020-1-23',"%Y-%m-%d")
    end_ori = datetime.datetime.strptime('2022-7-31',"%Y-%m-%d")
    start_date = datetime.datetime.strptime('2020-3-1',"%Y-%m-%d")
    end_date = datetime.datetime.strptime('2022-7-31',"%Y-%m-%d")

    trend_flg = True if config['data']['trend_flg'] else False
    use_saint_graph = True
    graph_type = 'county' # TODO: county, state, hierCS

    saint_batch_size = 600
    saint_sample_type = 'random_walk'
    saint_walk_length = 2
    saint_shuffle_order = 'node_first'

    ## GIB

    model_type='GAT'
    num_features=1
    num_classes=1
    normalize=True
    reparam_mode = config['model']['reparam_mode']
    prior_mode= config['model']['prior_mode']
    latent_size=16
    sample_size=1
    num_layers=2
    # struct_dropout_mode=("DNsampling","Bernoulli",0.1,0.5,"norm",2)
    struct_dropout_mode_dict = {0: ("Nsampling",'Bernoulli',0.1,0.5,"norm"), 1: ("DNsampling","Bernoulli",0.1,0.5,"norm",2), 2: ("standard",0.6)}
    struct_dropout_mode= struct_dropout_mode_dict[config['model']['struct_dropout_mode']]
    
    dropout=True,
    val_use_mean=True
    reparam_all_layers=(-2,)
    with_relu = True
    head = config['model']['head']

    # gsat
    learn_edge_att = True
    extractor_drop = 0.5
    use_edge_attr = True
    gib_drop = config['model']['gib_drop'] # defalt 0.3
    fix_r = False
    decay_interval = 10
    decay_r = 0.1
    final_r = 0.5
    init_r = 0.9

    # 
    d_model = 16
    hidden_size = 128
    d_model_temp = 128
    d_model_spat = 64
    only_spat = True if config['model']['only_spat'] else False
    beta1 = config['model']['beta1']
    beta2 = config['model']['beta2']

    # abl
    spat_gsat = True if config['model']['spat_gsat'] else False
    temp_gsat = True if config['model']['temp_gsat'] else False
    # try:
    #     dist_thr = config['dist_thr']
    # except:
    #     dist_thr = 2
    dist_thr = config.get('model').get('dist_thr', 10000)
    print('dist thr *************************', dist_thr)

    top_k_spat = 10
    top_k_temp = 20

args = ParamConfig()
logger = print_args(args)










