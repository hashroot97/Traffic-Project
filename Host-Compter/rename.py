import os


dir_name_empty = '.\\Data\\Main_empty\\'
list_dir = os.listdir(dir_name_empty)
for i in range(len(list_dir)):
    print(list_dir[i])
    os.rename(dir_name_empty+list_dir[i], dir_name_empty+str(i)+'_NoTraf'+'.jpg')

              

    
