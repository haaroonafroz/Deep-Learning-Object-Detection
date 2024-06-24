from yacs.config import CfgNode as CN
import yaml

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values for the project.
    Returns:
        CfgNode: Default configuration object.
    """
    _C = CN()

    # Data settings
    _C.DATA = CN()
    _C.DATA.DATASET = 'CISOL'
    _C.DATA.ROOT = './data/cisol_TD-TSR' # /kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR'

    # Model settings
    _C.MODEL = CN()
    _C.MODEL.NUM_CLASSES = 6
    _C.MODEL.BACKBONE = 'resnet_50'

    # Training settings
    _C.TRAIN = CN()
    _C.TRAIN.BASE_LR = 0.0005 
    _C.TRAIN.MILESTONES = [10, 20]
    _C.TRAIN.GAMMA = 0.1
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.NUM_EPOCHS = 30
    _C.TRAIN.EARLY_STOPPING = False

    # Augmentation settings
    _C.AUGMENTATION = CN()
    _C.AUGMENTATION.HORIZONTAL_FLIP_PROB = 0.3
    _C.AUGMENTATION.ROTATION_DEGREES = 10

    # Miscellaneous settings
    _C.MISC = CN()
    _C.MISC.RUN_NAME = 'default_run'
    _C.MISC.RESULTS_CSV = '/kaggle/working/repository_content/results'
    _C.MISC.SAVE_MODEL_PATH = '/kaggle/working/repository_content/saved_models'
    _C.MISC.PRETRAINED_WEIGHTS = ''
    _C.MISC.FROZEN_LAYERS = []
    _C.MISC.NO_CUDA = False

    return _C.clone()

def get_cfg_from_file(cfg_file):
    """
    Load configuration from a file.
    Args:
        cfg_file (str): Path to the configuration file.
    Returns:
        CfgNode: Configuration object.
    """
    with open(cfg_file, 'r') as file:
        cfg_dict = yaml.safe_load(file)

    cfg = get_cfg_defaults()
    cfg.merge_from_other_cfg(CN(cfg_dict))
    cfg.freeze()
    return cfg