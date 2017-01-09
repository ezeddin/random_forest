import numpy as np

ECOLI_PATH = ('../data/0.Ecoli/Xy.txt', np.float)
GLASS_PATH = ('../data/1.Glass/Xy.txt', np.float)
LIVER_PATH = ('../data/2.Liver/Xy.txt', np.float)
LETTERS_PATH = ('../data/3.Letters/Xy.txt', np.float)
SAT_IMAGES_PATH = ('../data/4.Sat Images/Xy.txt', np.float)
WAVEFORM_PATH = ('../data/5.Waveform/Xy.txt', np.float)
IONOSPHERE_PATH = ('../data/6.Ionosphere/Xy.txt', np.float)
DIABETES_PATH = ('../data/7.Diabetes/Xy.txt', np.float)
SONER_PATH = ('../data/8.Sonar/Xy.txt', np.float)
BREAST_CANCER_PATH = ('../data/9.Breast Cancer/Xy.txt', np.float)
VOTES_PATH = ('../data/10.Votes/Xy.txt', np.float)
VEHICLE_PATH = ('../data/11.Vehicle/Xy.txt', np.float)
path_list = [ECOLI_PATH, GLASS_PATH, LIVER_PATH, LETTERS_PATH, SAT_IMAGES_PATH, WAVEFORM_PATH, IONOSPHERE_PATH, DIABETES_PATH, SONER_PATH, BREAST_CANCER_PATH,VOTES_PATH,VEHICLE_PATH]

def parse_dataset(path_list):
    return [np.genfromtxt(data_path, delimiter=',', dtype=data_type) for data_path, data_type in path_list]

def get_datasets():
    return parse_dataset(path_list)


