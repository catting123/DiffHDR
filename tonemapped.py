import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import *
from utils.matric import *

def tonemapped(input_dir, output_dir, mode=1, suffix='.hdr'):
    files = get_all_files(input_dir, suffix=suffix)
    assert 1 <= mode <= 3, "mode is illegal!"
    factor = 3
    if mode == 1:
        tonemap_algorithm = cv2.createTonemapDrago()
    elif mode == 2:
        tonemap_algorithm = cv2.createTonemapReinhard()
        factor = 1
    else:
        tonemap_algorithm = cv2.createTonemapMantiuk()
    for file in files:
        read_path = os.path.join(input_dir, file)
        if suffix == ".hdr":
            image = read_hdr(read_path)
        else:
            image = cv2.imread(read_path).astype(np.float32)
        image_processed = tonemap_algorithm.process(image)
        image_processed *= factor
        write_path = os.path.join(output_dir, file)
        write_png(write_path[:-3]+"png", image_processed)


plt.rcParams.update({'font.size': 18})

def draw_images(title):
    f, axs = plt.subplots(2, 4, figsize=(20, 8))
    f.suptitle(title)
    for i in range(4):
        gt_img = cv2.imread(f"./image/GT/Tonemapped/0{i+1}.png")
        re_img = cv2.imread(f"./image/Restore/Tonemapped/0{i+1}.png")
        axs[0, i].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        axs[1, i].imshow(cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB))
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        title = "_".join(title.split())
    plt.savefig(f'./results/{title}.pdf')
    plt.show()
    
if __name__ == '__main__':
    GT_dir_origin = "./image/GT/Origin/"
    Restore_dir_origin = "./image/Restore/Origin/"
    Debevec_dir_origin = "./image/Debevec/Origin/"


    GT_dir_tonemapped = "./image/GT/Tonemapped/"
    Restore_dir_tonemapped  = "./image/Restore/Tonemapped/"
    Debevec_dir_tonemapped = "./image/Debevec/Tonemapped/"
    
    
    tonemapped(GT_dir_origin, GT_dir_tonemapped)
    tonemapped(Restore_dir_origin, Restore_dir_tonemapped)
        
    draw_images("Drago Tonemap")