import matplotlib.pyplot as plt
import glob
import cv2, os
from label import get_label
import numpy as np

def data_gain():
  """

  files : 生成した画像のパスを格納したリスト
  list  : {accession : class(label)}

  """
  files = glob.glob('/home/nakanishi/sotuken/machine-genome-classification/data/img/*.png')
  list = get_label('~/sotuken/machine-genome-classification/data/csv/creatures.csv')
  
  file_list = []
  tmp = []
  label_list = []
  
  for file in files:
    file = os.path.split(file)[1].replace('.png', '')
    tmp.append(file)
    
  num = len(files)
  num_file = num
  a = b = c = 0
  

  for n in range(num):

    if tmp[n] not in list:
      num_file -= 1
      continue
    """
    if list[tmp[n]] == 0:
      a = a + 1
        
    if list[tmp[n]] == 1:
      b = b + 1
        
    if list[tmp[n]] == 2:
      c = c + 1
    """
    cls = list[tmp[n]]
    label_list.append(cls)
    file = cv2.imread(files[n])
    file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    file_list.append(file)

  """
  for i in range(192):
    for j in range(192):
      print(file_list[0][i][j], end="\t")
    print (end="\n")
  """
  
  return file_list, label_list