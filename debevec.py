import cv2
from utils.utils import *
from utils.matric import *
import os
import numpy as np



def weight(z, zmin=0, zmax=255):
    zmid = (zmin+zmax) / 2
    return z - zmin if z <= zmid else zmax - z

def color_split(images):
    images_b, images_g, images_r = [], [], []
    for image in images:
        b, g, r = cv2.split(image)
        images_b.append(b)
        images_g.append(g)
        images_r.append(r)
    return images_b, images_g, images_r

# Firstly, randomly select N points
def hdr_debevec(images, times, l, sample_nums):
    w = [weight(z) for z in range(256)]
    B = np.log(times)
    Z = []

    n, m = images[0].shape
    step = n*m // sample_nums
    sample_indices = np.arange(0, n*m, step, dtype=np.int32)
    sx, sy = sample_indices // m, sample_indices % m
    for img in images:
        tmp = []
        for x, y in zip(sx,sy):
            tmp.append(img[x][y])
        Z.append(tmp)
    
    # samples = [(random.randint(0, images[0].shape[0]-1), random.randint(0, images[0].shape[1]-1)) for i in range(sample_nums)]
    # for img in images:
    #     Z += [[img[r[0]][r[1]] for r in samples]]
    
    Z = np.array(Z).T
    return response_curve_solver(Z, B, l, w)

# Secondly, generate the matrix A and b
# Thirdly, solve the linear system using SVD
def response_curve_solver(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(Z.shape[0]*Z.shape[1]+n+1, n+Z.shape[0]), dtype=np.float32)
    b = np.zeros(shape=(A.shape[0], 1), dtype=np.float32)
    
    # Include the data-fitting equations
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = int(Z[i][j])
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij * B[j]
            k += 1
    
    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1
    
    # Include the smoothness equations
    for i in range(n-1):
        A[k][i] = l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] = l*w[i+1]
        k += 1
    
    # Solve the system using SVD
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:n]
    lE = x[n:]
    
    return g, lE

# Finally, return the crf, which should be a numpy array of shape (3, 256) as we have 3 channels
def get_crf(images, times, l = 50):
    sample_nums = int(np.ceil(255*2 / (len(times)-1)) * 2) 
    images_b, images_g, images_r = color_split(images)
    g_b, lE_b = hdr_debevec(images_b, times, l, sample_nums)
    g_g, lE_g = hdr_debevec(images_g, times, l, sample_nums)
    g_r, lE_r = hdr_debevec(images_r, times, l, sample_nums)
    return [g_b, g_g, g_r]

def minmax_scaler(data):
    minimum = np.min(data)
    maximum = np.max(data)
    return (data - minimum) / (maximum - minimum)


def get_single_map(images, times, g):
    radiance_map = np.zeros((images[0].shape[0], images[0].shape[1]))
    sum_down = np.zeros((images[0].shape[0], images[0].shape[1]))
    w = np.array([weight(z) for z in range(256)])

    for k in range(len(times)):
        Zij = images[k]
        Wij = np.maximum(w[Zij], 1)
        radiance_map += Wij*(g[Zij][:,:,0]-np.log(times[k]))
        sum_down += Wij
    radiance_map = radiance_map / sum_down
    return radiance_map

def get_radiance_map(images, times, crf):
    images_b, images_g, images_r = color_split(images)
    radiance_map = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.float32)

    radiance_map[:, :, 0] = get_single_map(images_b, times, crf[0])
    radiance_map[:, :, 1] = get_single_map(images_g, times, crf[1])
    radiance_map[:, :, 2] = get_single_map(images_r, times, crf[2])
    radiance_map = minmax_scaler(radiance_map)
    return radiance_map
