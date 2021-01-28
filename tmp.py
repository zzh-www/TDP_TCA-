import os
import pandas as pd
# def class_map(x):
#     if x == "b'buggy'":
#         return  1
#     elif x == "b'clean'" :
#         return  0
#     else:
#         return x
#
# def isD_map(x):
#     if x == "b'TRUE'":
#         return 1
#     elif x== "b'FALSE'":
#         return 0
#     else:
#         return x
# for dir_path, dir_names, file_names in os.walk('data/csvs'):
#     print(dir_path)
#     files = []
#     for file in file_names:
#         if file.endswith('.csv'):
#             file = file.replace('.csv','')
#             files.append(file)
#
#     with open(dir_path.replace('/','_').replace('\\','_')+'.txt', 'w') as file_obj:
#         print(files)
#         # create s&t txt
#         for i in range(len(files)-1):
#             for j in range(i+1,len(files)):
#                 file_obj.write(str(files[i])+','+str(files[j])+'\n')
#     # fix AEEEM , Relink csv
#     if dir_path.endswith('AEEEM'):
#         for file in files:
#             df = pd.read_csv(dir_path.replace('\\','/')+'/'+file+'.csv')
#             # # ①使用字典进行映射
#             # data["gender"] = data["gender"].map({"男": 1, "女": 0})
#             # ​
#             #
#             # # ②使用函数
#             # def gender_map(x):
#             #     gender = 1 if x == "男" else 0
#             #     return gender
#             # 注意这里传入的是函数名，不带括号
#             # data["gender"] = data["gender"].map(gender_map)
#             print(df['class'].tolist())
#             df['class'] = df['class'].map(class_map)
#             print(df['class'].tolist())
#             df.to_csv(dir_path.replace('\\','/')+'/'+file+'.csv',index=False)
#     elif dir_path.endswith('Relink'):
#         for file in files:
#             df = pd.read_csv(dir_path.replace('\\', '/') + '/' + file + '.csv')
#             print(df['isDefective'].tolist())
#             df['isDefective'] = df['isDefective'].map(isD_map)
#             print(df['isDefective'].tolist())
#             df.to_csv(dir_path.replace('\\', '/') + '/' + file + '.csv',index=False)


import numpy as np
arrs=[]
arr_fs = []
arr = np.array([[1, 1, 1], [65, 3, 23424 ],[0,0,0],[0,0,0]])
arr_f = np.array([[0],[1],[0],[1]])
for i in range(len(arr)):
    if arr[i].tolist() !=  np.zeros(len(arr[0])).tolist():
        arrs.append(arr[i])
        arr_fs.append(arr_f[i])
arr = np.array(arrs)
arr_f = np.array(arr_fs)
# arr =
# arr.delete(arr[arr == [1,1,1]])
# np.delete(arr,arr[arr==1,1,1])
# arr = arr[arr.tolist() != [0,0,0]]
# arr = np.delete(arr,arr == np.array([0,0,0]))
print(arr)
print(arr_f)