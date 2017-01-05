import numpy as np

ECOLI_PATH = ('../data/0.Ecoli/ecoli.data.txt', None)
GLASS_PATH = ('../data/1.Glass/glass.data', np.float)
LIVER_PATH = ('../data/2.Liver/bupa.data', np.float)
LETTERS_PATH = ('../data/3.Letters/letter-recognition.data.txt', None)
SAT_IMAGES_PATH = ('../data/4.Sat Images/sat.all', np.float)
WAVEFORM_PATH = ('../data/5.Waveform/waveform-+noise.data', np.float)
IONOSPHERE_PATH = ('../data/6.Ionosphere/ionosphere.data.txt', None)
DIABETES_PATH = ('../data/7.Diabetes/pima-indians-diabetes.data', np.float)
SONER_PATH = ('../data/8.Sonar/sonar.all-data.txt', None)
BREAST_CANCER_PATH = ('../data/9.Breast Cancer/breast-cancer-wisconsin.data.txt', np.float)
path_list = [ECOLI_PATH, GLASS_PATH, LIVER_PATH, LETTERS_PATH, SAT_IMAGES_PATH, WAVEFORM_PATH, IONOSPHERE_PATH, DIABETES_PATH, SONER_PATH, BREAST_CANCER_PATH]

def parse_dataset(path_list):
    return [np.genfromtxt(data_path, delimiter=',', dtype=data_type) for data_path, data_type in path_list]

