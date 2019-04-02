import os

dir_path = "/home/sandbox/Scrivania/filippoaleotti/full_kitti"
total = 0
counter = 0
with open("./filenames/kitti_train_files.txt",'r') as kitti:
    for l in kitti:
        l1,l2 = l.split(' ') 
        #l1 = l1.replace('jpg','png').strip()[0]
        #l2 = l2.replace('jpg','png')
        l1 = os.path.join(dir_path,l1)
        l2 = os.path.join(dir_path,l2)
        l1 = l1.strip()
        l2 = l2.strip()
        total += 1
        if not os.path.exists(l1):
            print("{} doesn't exist".format(l1))
            continue

        if not os.path.exists(l2):
            print("{} doesn't exist".format(l2))
            continue

        counter += 1
print("{} of {}".format(counter, total))