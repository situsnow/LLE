
AMSRV = 'amazon_us_reviews'

DISTILBERT = 'distilBERT'
BERT = 'BERT'

HUGGINGFACE_SPLIT_PERCENT = '%'
HUGGINGFACE_SPLIT_ABS = 'abs'

CUDA = 'cuda'
CPU = 'cpu'

CHECKPOINT_PREFIX = 'checkpoint'

OCCLUSION = 'occlusion'
LIME = 'lime'
KERNEL_SHAP = 'kernel_shap'
DEEP_SHAP = 'deep_shap'
GRADIENT = 'gradient'
LRP = 'lrp'

RANK = 'rank'
MULTILABEL = 'multilabel'

POS = 'pos'
NEG = 'neg'

BERT_COS = 'bert-cos'
N_GRAM = 'n-gram'

# =============================
FS_TRAIN_SRC = 'fs_train_src.exp'
FS_TRAIN_TGT = 'fs_train_tgt.exp'
FS_DEV_SRC = 'fs_dev_src.exp'
FS_DEV_TGT = 'fs_dev_tgt.exp'
FS_TEST_SRC = 'fs_test_src.exp'
FS_TEST_TGT = 'fs_test_tgt.exp'
FS_MEM_SRC = 'fs_mem_src.exp'   # for concated memory of all previous tasks but store in current folder
FS_MEM_TGT = 'fs_mem_tgt.exp'
FS_INF_SRC = 'fs_inf_src.exp'   # for experiment result analysis
FS_INF_TGT = 'fs_inf_tgt.exp'


TRAIN = 'train'    # for train split
DEV = 'dev'     # for dev split
TEST = 'test'   # for test split
MEM = 'mem'    # for replaying latest memory (regenerate the explanation)
INF = 'inf'    # for evaluation
EXP = 'exp'   # for expired memory


SRC = "src"
TGT = 'tgt'

FAIRSEQ_FOLDER = 'am_fs'
BBOX_FOLDER = 'bert_base_models'

HEURISTICS_RANDOM = 'random'
HEURISTICS_SIM = 'sim'
HEURISTICS_NOMEM = 'nomem'
HEURISTICS_OLDMEM= 'oldmem'

FAIRSEQ_RAN_CHECKPOINTS = 'checkpoints_withRanMemory'
FAIRSEQ_SIM_CHECKPOINTS = 'checkpoints_withSimMemory'
FAIRSEQ_NOMEM_CHECKPOINTS = 'checkpoints_withNoMemory'
FAIRSEQ_OLDMEM_CHECKPOINTS = 'checkpoints_withOldMemory'
LAST_CHECKPOINT = 'checkpoint_last.pt'
COPY_CHECKPOINT = 'checkpoint_copy.pt'

FAIRSEQ_RAN_OUTPUT = 'stdout.txt'
FAIRSEQ_SIM_OUTPUT = 'stdout_sim.txt'
FAIRSEQ_NOMEM_OUTPUT = 'stdout_nomem.txt'
FAIRSEQ_OLDMEM_OUTPUT = 'stdout_oldmem.txt'

FAIRSEQ_RAN_PARAMETERS = 'sequential_explainer_parameters'
FAIRSEQ_OLDMEM_PARAMETERS = 'sequential_explainer_parameters_oldmem'
FAIRSEQ_NOMEM_PARAMETERS = 'sequential_explainer_parameters_nomem'

HEURISTICS_CHECKPOINT_DICT = {HEURISTICS_RANDOM: FAIRSEQ_RAN_CHECKPOINTS,
                              HEURISTICS_NOMEM: FAIRSEQ_NOMEM_CHECKPOINTS,
                              HEURISTICS_OLDMEM: FAIRSEQ_OLDMEM_CHECKPOINTS}

FAIRSEQ_OUTPUT_DICT = {HEURISTICS_RANDOM: FAIRSEQ_RAN_OUTPUT,
                       HEURISTICS_NOMEM: FAIRSEQ_NOMEM_OUTPUT, HEURISTICS_OLDMEM: FAIRSEQ_OLDMEM_OUTPUT}

FAIRSEQ_PARAMETERS_DICT = {HEURISTICS_RANDOM: FAIRSEQ_RAN_PARAMETERS,
                           HEURISTICS_OLDMEM: FAIRSEQ_OLDMEM_PARAMETERS,
                           HEURISTICS_NOMEM: FAIRSEQ_NOMEM_PARAMETERS}

REVERSE = 'reverse'
DIRECT = 'direct'
RANDOM = 'random'
AMAZON_RANDOM = 'amazon_random'
AMAZON_REVERSE = 'amazon_reverse'

MEM_GRP_CRTR_CONFIG = 'config'
MEM_GRP_CRTR_LBL = 'label'

AMAZON_COLUMNS = ['star_rating', 'review_body']

MEMORY_FOLDER = 'memory'
MEMORY_FILE = 'memory.csv'

PGM_OPTION_SIMILARITY = 'sim'
PGM_OPTION_IOU = 'iou'

PRETRAINED_BERT_FOLDER = 'transformers_models'

MEMORY_RANDOM = 'random'
MEMORY_CLUSTER = 'cluster'
MEMORY_GSS = 'gradient'

MEMORY_CLUSTER_DENSITY = 'density'



