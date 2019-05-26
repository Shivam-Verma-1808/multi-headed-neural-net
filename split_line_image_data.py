import glob
import os
import random
from PIL import Image, ImageDraw
import sys

if len(sys.argv) < 3 :
    print("Usage: python split_line_image_data.py path_in/ path_out/")
    print("Example: python split_line_image_data.py /home/user/Downloads/sem_6/CS_671_DL/Assignment_1/github/cla_test/ /home/user/Downloads/sem_6/CS_671_DL/Assignment_2/test/")
    sys.exit (1)
path_in= str(sys.argv[1])
path_out= str(sys.argv[2])

path_out_train = path_out + 'train/'
os.makedirs(path_out_train)

path_out_test = path_out + 'test/'
os.makedirs(path_out_test)



arr = random.sample(range(1000),600)
print(arr)
for dir in glob.glob(path_in+'class_*') :
    print(dir)
    for image_file in glob.glob(dir+'/'+'*.jpg') :
        print(image_file)
        image_id = str(image_file).split('/')[-1]
        print(image_id)
        image_number = image_id.split('_')[-1].split('.')[0]
        if(int(image_number) in arr) :
            print(image_number,'mickey_mouse')
            output_path = path_out_train
        else :
            output_path = path_out_test
        img = Image.open(image_file)
        img.save(output_path + image_id)
        img.close()

