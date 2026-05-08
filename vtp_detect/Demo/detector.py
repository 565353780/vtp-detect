import os
import cv2
import torch
import numpy as np

from tqdm import trange

from vtp_detect.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    vhome = '/vepfs-cnbja62d5d769987/lichanghao'

    model_file_path = f'{vhome}/chLi/Model/VTP/VTP-Large-f16d64'
    dtype = "auto"
    device = "cuda:0"

    image_file_path = f'{home}/tmp/test.png'
    if not os.path.exists(image_file_path):
        H, W = 512, 512  # 可以根据实际需要调整大小
        noise_img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
        cv2.imwrite(image_file_path, noise_img)

    detector = Detector(model_file_path, dtype, device)

    for _ in trange(10):
        vtp_feature = detector.detect(
            torch.rand([3, 512, 512, 3], dtype=torch.float32, device="cpu")
        )

    print("vtp_feature:")
    print("cls_token:", vtp_feature["cls_token"].shape)
    print("patch_tokens:", vtp_feature["patch_tokens"].shape)

    for _ in trange(10):
        vtp_feature = detector.detectFile(image_file_path)

    print("vtp_feature:")
    print("cls_token:", vtp_feature["cls_token"].shape)
    print("patch_tokens:", vtp_feature["patch_tokens"].shape)
    return True



if __name__ == "__main__":
    demo()
