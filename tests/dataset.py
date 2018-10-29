import pathlib


TEST_ROOT = pathlib.Path(__file__).parent
DATASET_ROOT = TEST_ROOT/'data'
CLB_ROOT = DATASET_ROOT/'cmu_us_clb_arctic'
SLT_ROOT = DATASET_ROOT/'cmu_us_slt_arctic'

CLB_DIR = CLB_ROOT/'wav'
CLB_WAV = CLB_DIR/'arctic_a0001.wav'
CLB_WAV2 = CLB_DIR/'arctic_a0002.wav'

SLT_DIR = SLT_ROOT/'wav'
SLT_WAV = SLT_DIR/'arctic_a0001.wav'
SLT_WAV2 = SLT_DIR/'arctic_a0002.wav'
