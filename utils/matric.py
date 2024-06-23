import torch
from pytorch_fid import fid_score
import lpips
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dims = 2048
batch_size = 128


def calc_fid_from_folder(images_folder1, images_folder2):
    fid = fid_score.calculate_fid_given_paths(
        [images_folder1, images_folder2],
        batch_size=batch_size,
        dims=dims,
        device=device,
    )
    return fid


def calc_lpips(images1, images2):
    if type(images1) != list:
        assert type(images1) == type(images1) == np.ndarray
        images1 = [images1]
        images2 = [images2]

    assert len(images1) == len(images2)
    lpips_model = lpips.LPIPS(net="alex", version=0.1).to(device)
    lpips_scores = []
    for i in tqdm(range(len(images1)), disable=len(images1) == 1):
        image1 = lpips.im2tensor(images1[i]).to(device)
        image2 = lpips.im2tensor(images2[i]).to(device)
        lpips_score = lpips_model(image1, image2).detach().cpu().numpy()
        lpips_scores.append(lpips_score)
    return np.mean(lpips_scores)


def calc_lpips_from_folder(images_folder1, images_folder2):
    images1 = read_images(images_folder1, suffix=".png")
    images2 = read_images(images_folder2, suffix=".png")
    return calc_lpips(images1, images2)


def calc_ssim(images1, images2):
    if type(images1) != list:
        assert type(images1) == type(images1) == np.ndarray
        images1 = [images1]
        images2 = [images2]

    assert len(images1) == len(images2)
    ssim_scores = []
    for i in tqdm(range(len(images1)), disable=len(images1) == 1):
        image1 = images1[i]
        image2 = images2[i]
        ssim_score = ssim(
            image1,
            image2,
            win_size=11,
            data_range=255,
            gradient=False,
            channel_axis=-1,
            device=device,
        )
        ssim_scores.append(ssim_score)
    return np.mean(ssim_scores)


def calc_ssim_from_folder(images_folder1, images_folder2):
    images1 = read_images(images_folder1, suffix=".png")
    images2 = read_images(images_folder2, suffix=".png")
    return calc_ssim(images1, images2)


def calc_matrics(images_folder1, images_folder2, matric):
    if matric.lower() == "fid":
        calc_fn = calc_fid_from_folder
    elif matric.lower() == "lpips":
        calc_fn = calc_lpips_from_folder
    elif matric.lower() == "ssim":
        calc_fn = calc_ssim_from_folder
    else:
        raise Exception("matric not found, matric must in ('fid','lpips')")

    matric_score = calc_fn(images_folder1, images_folder2)
    return matric_score


