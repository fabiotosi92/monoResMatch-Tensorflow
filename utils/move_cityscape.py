import os
import shutil
from tqdm import tqdm


left_dir = "/home/sandbox/Scrivania/filippoaleotti/cityscape/leftImg8bit/train_extra"
right_dir = "/home/sandbox/Scrivania/filippoaleotti/cityscape/rightImg8bit/train_extra"

dest_folder = "/home/sandbox/Scrivania/filippoaleotti/cityscape/"
with open('./filenames/cityscapes_train_files.txt','r') as f:
    lines = f.readlines()

do_move = True
if do_move:
    with tqdm(total = len(lines)) as bar:
        for line in lines:
            left,right = line.strip().split()
            source_left = os.path.join(left_dir, left)
            source_right = os.path.join(right_dir, right)
            dest_left = os.path.join(dest_folder, left)
            dest_right = os.path.join(dest_folder, right)
            # left and right imgs are inside a folder with city name
            # if that folder doesn't exist move command won't work
            dest_fold = os.path.join(dest_folder, os.path.dirname(left))
            if not os.path.exists(dest_fold):
                #left and right share same folder
                os.mkdir(dest_fold)
            if os.path.exists(source_left) and os.path.exists(source_right): 
                shutil.move(source_left, dest_left)
                shutil.move(source_right, dest_right)
            else:
                if not os.path.exists(source_left):
                    assert not os.path.exists(source_right)
                if not os.path.exists(source_right):
                    assert not os.path.exists(source_left)
            bar.update(1)

do_check = True

if do_check:
    print ("final check...")
    with tqdm(total = len(lines)) as bar:
        for line in lines:
            left,right = line.strip().split()
            dest_left = os.path.join(dest_folder, left)
            dest_right = os.path.join(dest_folder, right)
            try:
                assert os.path.exists(dest_left)
            except AssertionError:
                print("Not found: {}".format(dest_left))
            try:
                assert os.path.exists(dest_right)
            except AssertionError:
                print("Not found: {}".format(dest_right))
            print("{} {} exists".format(dest_left, dest_right))
            bar.update(1)

print("Done!")