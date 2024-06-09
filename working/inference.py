DEBUG = True

# [
#     "submission",
#     "church",
#     "dioscuri",
#     "lizard",
#     "multi-temporal-temple-baalshamin",
#     "pond",
#     "transp_obj_glass_cup",
#     "transp_obj_glass_cylinder",
# ]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--validation", action="store_true")
args = parser.parse_args()

if args.validation:
    DATA_TYPE = "train"
    SCENE_TYPE = "submission"
    #SCENE_TYPE = "transp_obj_glass_cup"
    #SCENE_TYPE = "church"
else:
    DATA_TYPE = "test"
    SCENE_TYPE = "submission"

from pathlib import Path
output_dir = Path("/kaggle/working/")

import os
import shutil

os.makedirs("/root/.cache/torch/hub/checkpoints/", exist_ok=True)
shutil.copy(
    "/kaggle/input/aliked-pytorch-aliked-n16-v1/aliked-n16.pth",
    "/root/.cache/torch/hub/checkpoints/",
)
shutil.copy(
    "/kaggle/input/lightglue-pytorch-aliked-v1/aliked_lightglue.pth",
    "/root/.cache/torch/hub/checkpoints/",
)
shutil.copy(
    "/kaggle/input/lightglue-pytorch-aliked-v1/aliked_lightglue.pth",
    "/root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth",
)
shutil.copy(
    "/kaggle/input/check-orientation/2020-11-16_resnext50_32x4d.zip",
    "/root/.cache/torch/hub/checkpoints/",
)
shutil.copy(
    "/kaggle/input/dinov2-repo/dinov2_vits14_pretrain.pth",
    "/root/.cache/torch/hub/checkpoints/",
)
shutil.copy(
    "/kaggle/input/dinov2-repo/dinov2_vits14_voc2012_linear_head.pth",
    "/root/.cache/torch/hub/checkpoints/",
)

import argparse
import cv2
import gc
import h5py
import importlib
import itertools
import kornia as K
import kornia.feature as KF
import math
import mmcv
import numpy as np
import pandas as pd
import pycolmap
import random
import shutil
import sys
import torch
import torch.nn.functional as F

from check_orientation.pre_trained_models import create_model
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from kneed import KneeLocator
from lightglue import ALIKED
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image as T_read_image
from torchvision.io import ImageReadMode
from torchvision import transforms as T
from tqdm.auto import tqdm

##########################################
# DINOv2 Segmenter
# https://github.com/facebookresearch/dinov2

sys.path.append('/kaggle/input/dinov2-repo/dinov2')
import dinov2.eval.segmentation.models

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model

def dinov2_segmentation(
    paths,
    feature_dir,
    device_id,
):

    device = torch.device(f"cuda:{device_id}")

    backbone_model = torch.hub.load('/kaggle/input/dinov2-repo/dinov2', model="dinov2_vits14", source='local')
    backbone_model.eval()
    backbone_model.to(device)

    #head_config_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_linear_config.py"
    cfg = mmcv.Config.fromfile("/kaggle/input/dinov2-repo/dinov2_vits14_voc2012_linear_config.py")

    model = create_segmenter(cfg, backbone_model=backbone_model)
    #head_checkpoint_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_linear_head.pth"
    load_checkpoint(model, "/kaggle/input/dinov2-repo/dinov2_vits14_voc2012_linear_head.pth", map_location="cpu")
    model.to(device)
    model.eval()

    with h5py.File(feature_dir / f"fg_mask{device_id}.h5", mode="w") as f_fg_mask:
        for img_path in tqdm(paths, dynamic_ncols=True, desc=f"Segmenting[GPU{device_id}]"):
            image = cv2.imread(str(img_path))

            segmentation_logits = inference_segmentor(model, image)[0]
            mask = np.zeros_like(segmentation_logits)
            mask[segmentation_logits == 5] = 255

            key = img_path.name
            f_fg_mask[key] = mask

    return

def merge_single_h5(
    feature_dir,
    input_name_list,
    output_name,
):
    with h5py.File(feature_dir / output_name, mode="w") as f_output:

        for input_name in input_name_list:
            with h5py.File(feature_dir / input_name, mode="r") as f_input:
                for key in f_input.keys():
                    f_output[key] = f_input[key][...]

    return

def merge_double_h5(
    feature_dir,
    input_name_list,
    output_name,
):
    with h5py.File(feature_dir / output_name, mode="w") as f_output:

        for input_matches_name in input_name_list:
            with h5py.File(feature_dir / input_matches_name, mode="r") as f_input:
                for key1 in f_input.keys():
                    group  = f_output.require_group(key1)
                    for key2 in f_input[key1].keys():
                        group.create_dataset(key2, data=f_input[key1][key2][...])

    return

# DINOv2 Segmenter
##########################################

##########################################
# Detect keypoints by ALIKED

def pad_to_square(image):
    if len(image.shape) == 3:
        height, width, _ = image.shape
        max_dim = max(height, width)
        padded_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        padded_image[:height, :width, :] = image
    elif len(image.shape) == 2:
        height, width = image.shape
        max_dim = max(height, width)
        padded_image = np.zeros((max_dim, max_dim), dtype=np.uint8)
        padded_image[:height, :width] = image
    else:
        raise ValueError("Invalid image shape")
    return padded_image

# rotate image 90 degrees clockwise
def detect_keypoints(
    paths,
    feature_dir,
    device_id,
    num_features,
    detection_threshold,
    resize_to,
):
    device = torch.device(f"cuda:{device_id}")

    dtype = torch.float32 # ALIKED has issues with float16

    extractor = ALIKED(
        max_num_keypoints=num_features,
        detection_threshold=detection_threshold,
        resize=resize_to,
    ).eval().to(device, dtype)

    with h5py.File(feature_dir / f"keypoints_deg{device_id}.h5", mode="w") as f_keypoints_deg, \
         h5py.File(feature_dir / f"descriptors_deg{device_id}.h5", mode="w") as f_descriptors_deg, \
         h5py.File(feature_dir / f"offsets{device_id}.h5", mode="w") as f_offsets_deg, \
         h5py.File(feature_dir / f"keypoints{device_id}.h5", mode="w") as f_keypoints:

        for path in tqdm(paths, desc=f"Computing keypoints[GPU{device_id}]", dynamic_ncols=True):

            _image = cv2.imread(str(path))
            _image = pad_to_square(_image)

            key = path.name

            f_keypoints_deg.create_group(key)
            f_descriptors_deg.create_group(key)
            f_offsets_deg.create_group(key)

            keypoints_list = []
            offset = 0
            for deg in ["0deg", "90deg", "180deg", "270deg"]:
                if deg != "0deg":
                    # rotate 90 degrees
                    _image = cv2.rotate(_image, cv2.ROTATE_90_CLOCKWISE)
                image = K.image_to_tensor(_image, False).float() / 255.
                image = K.color.bgr_to_rgb(image).to(device).to(dtype)

                with torch.inference_mode():
                    features = extractor.extract(image)

                keypoints = features["keypoints"].squeeze().detach().cpu().numpy()
                f_keypoints_deg[key][deg] = keypoints
                f_descriptors_deg[key][deg] = features["descriptors"].squeeze().detach().cpu().numpy()
                f_offsets_deg[key][deg] = offset
                offset += keypoints.shape[0]

                # rotate back the keypoints
                temp = keypoints.copy()
                if deg == "90deg":
                    keypoints[:, 0] = temp[:, 1]
                    keypoints[:, 1] = _image.shape[1] - temp[:, 0]
                elif deg == "180deg":
                    keypoints[:, 0] = _image.shape[1] - temp[:, 0]
                    keypoints[:, 1] = _image.shape[0] - temp[:, 1]
                elif deg == "270deg":
                    keypoints[:, 0] = _image.shape[0] - temp[:, 1]
                    keypoints[:, 1] = temp[:, 0]

                keypoints_list.append(keypoints)

            f_keypoints[key] = np.concatenate(keypoints_list)

    return

# special function for transparent object
def detect_keypoints_transparent(
    paths,
    feature_dir,
    device_id,
    num_features,
    detection_threshold,
    resize_to,
):
    device = torch.device(f"cuda:{device_id}")

    dtype = torch.float32 # ALIKED has issues with float16

    extractor = ALIKED(
        max_num_keypoints=num_features,
        detection_threshold=detection_threshold,
        resize=resize_to,
    ).eval().to(device, dtype)

    num_grids = 0

    def mask_keypoints(mask_area, keypoints):
        fg_idx = []
        for kpt in keypoints.astype(int):
            flag = (mask_area[kpt[1], kpt[0]] > 0)
            fg_idx.append(flag)
        return np.array(fg_idx)

    with h5py.File(feature_dir / "fg_mask.h5", mode="r") as f_fg_mask, \
         h5py.File(feature_dir / f"descriptors_grid{device_id}.h5", mode="w") as f_descriptors_grid, \
         h5py.File(feature_dir / f"keypoints_grid{device_id}.h5", mode="w") as f_keypoints_grid, \
         h5py.File(feature_dir / f"offsets_grid{device_id}.h5", mode="w") as f_offsets_grid, \
         h5py.File(feature_dir / f"keypoints{device_id}.h5", mode="w") as f_keypoints:

        for path in tqdm(paths, desc=f"Computing keypoints[GPU{device_id}]", dynamic_ncols=True):

            _image = cv2.imread(str(path))
            _mask = f_fg_mask[path.name][...]
            height, width = _image.shape[:2]
            step = 1024

            key = path.name

            f_descriptors_grid.create_group(key)
            f_keypoints_grid.create_group(key)
            f_offsets_grid.create_group(key)

            grid_id = 0
            offset = 0
            keypoints_list = []
            for y in range(0, height, step):
                for x in range(0, width, step):

                    mask = _mask[y:y+step, x:x+step]
                    if mask.sum() == 0:
                        grid_id += 1
                        continue

                    image = K.image_to_tensor(_image[y:y+step, x:x+step], False).float() / 255.
                    image = K.color.bgr_to_rgb(image).to(device).to(dtype)

                    with torch.inference_mode():
                        features = extractor.extract(image)

                    keypoints = features["keypoints"].detach().cpu().numpy().reshape(-1, 2)
                    features = features["descriptors"].detach().cpu().numpy().reshape(-1, 128)

                    # mask out keypoints that are not in the foreground
                    fg_idx = mask_keypoints(mask, keypoints)
                    keypoints = keypoints[fg_idx]
                    features = features[fg_idx]

                    if len(keypoints) == 0:
                        grid_id += 1
                        continue

                    f_keypoints_grid[key][f"{grid_id}"] = keypoints
                    f_descriptors_grid[key][f"{grid_id}"] = features
                    f_offsets_grid[key][f"{grid_id}"] = offset
                    grid_id += 1

                    offset += keypoints.shape[0]

                    keypoints[:, 0] += x
                    keypoints[:, 1] += y
                    keypoints_list.append(keypoints)

            f_keypoints[key] = np.concatenate(keypoints_list)

            num_grids = max(num_grids, grid_id)

    return num_grids

# Detect keypoints by ALIKED
##########################################

##########################################
# Match keypoints by LightGlue

# rotate image 90 degrees clockwise
def keypoint_distances(
    paths,
    index_pairs,
    feature_dir,
    device_id,
    early_stopping_thr,
    min_matches,
    verbose,
):
    device = torch.device(f"cuda:{device_id}")

    matcher_params = {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True,
    }
    matcher = KF.LightGlueMatcher("aliked", matcher_params).eval().to(device)

    with h5py.File(feature_dir / "keypoints_deg.h5", mode="r") as f_keypoints_deg, \
         h5py.File(feature_dir / "descriptors_deg.h5", mode="r") as f_descriptors_deg, \
         h5py.File(feature_dir / "offsets.h5", mode="r") as f_offsets_deg, \
         h5py.File(feature_dir / f"matches{device_id}.h5", mode="w") as f_matches:

            for idx1, idx2 in tqdm(index_pairs, desc=f"Computing keypoint distances[GPU{device_id}]", dynamic_ncols=True):
                key1, key2 = paths[idx1].name, paths[idx2].name

                keypoints1 = torch.from_numpy(f_keypoints_deg[key1]["0deg"][...]).to(device)
                descriptors1 = torch.from_numpy(f_descriptors_deg[key1]["0deg"][...]).to(device)

                best_deg = "0deg"
                max_matches = 0
                for deg in ["0deg", "90deg", "180deg", "270deg"]:

                    keypoints2 = torch.from_numpy(f_keypoints_deg[key2][deg][...]).to(device)
                    descriptors2 = torch.from_numpy(f_descriptors_deg[key2][deg][...]).to(device)
                    offset = f_offsets_deg[key2][deg][...]

                    with torch.inference_mode():
                        distances, indices = matcher(
                            descriptors1,
                            descriptors2,
                            KF.laf_from_center_scale_ori(keypoints1[None]),
                            KF.laf_from_center_scale_ori(keypoints2[None]),
                        )

                    n_matches = len(indices)
                    if verbose:
                        print(f"{key1}-{key2}: {n_matches} matches, rotation: {deg}")
                    if n_matches > max_matches:
                        best_deg = deg
                        max_matches = n_matches
                        best_indices = indices.detach().cpu().numpy().reshape(-1, 2)
                        best_indices[:, 1] += offset
                    if n_matches >= early_stopping_thr:
                        break

                # We have matches to consider
                if max_matches:
                    # Store the matches in the group of one image
                    if max_matches >= min_matches:
                        if verbose:
                            print(f"{key1}-{key2}: {max_matches} matches, best rotation: {best_deg}")
                        group  = f_matches.require_group(key1)
                        group.create_dataset(key2, data=best_indices)

    return

# special function for transparent object
def keypoint_distances_transparent(
    paths,
    index_pairs,
    feature_dir,
    num_grids,
    device_id,
    min_matches,
    verbose,
):
    device = torch.device(f"cuda:{device_id}")

    matcher_params = {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True,
    }
    matcher = KF.LightGlueMatcher("aliked", matcher_params).eval().to(device)

    with h5py.File(feature_dir / "keypoints_grid.h5", mode="r") as f_keypoints_grid, \
         h5py.File(feature_dir / "descriptors_grid.h5", mode="r") as f_descriptors_grid, \
         h5py.File(feature_dir / "offsets_grid.h5", mode="r") as f_offsets_grid, \
         h5py.File(feature_dir / f"matches{device_id}.h5", mode="w") as f_matches:

            for idx1, idx2 in tqdm(index_pairs, desc=f"Computing keypoint distances[GPU{device_id}]", dynamic_ncols=True):
                key1, key2 = paths[idx1].name, paths[idx2].name

                index_list = []
                for grid_id in range(num_grids):

                    if f_keypoints_grid[key1].get(f"{grid_id}") is None or f_keypoints_grid[key2].get(f"{grid_id}") is None:
                        continue

                    keypoints1 = torch.from_numpy(f_keypoints_grid[key1][f"{grid_id}"][...]).to(device)
                    descriptors1 = torch.from_numpy(f_descriptors_grid[key1][f"{grid_id}"][...]).to(device)
                    offset1 = f_offsets_grid[key1][f"{grid_id}"][...]

                    keypoints2 = torch.from_numpy(f_keypoints_grid[key2][f"{grid_id}"][...]).to(device)
                    descriptors2 = torch.from_numpy(f_descriptors_grid[key2][f"{grid_id}"][...]).to(device)
                    offset2 = f_offsets_grid[key2][f"{grid_id}"][...]

                    with torch.inference_mode():
                        distances, indices = matcher(
                            descriptors1,
                            descriptors2,
                            KF.laf_from_center_scale_ori(keypoints1[None]),
                            KF.laf_from_center_scale_ori(keypoints2[None]),
                        )

                    n_matches = len(indices)

                    # Store the matches in the group of one image
                    if n_matches >= min_matches:
                        if verbose:
                            print(f"grid_id1:{grid_id}, {key1}-{key2}: {n_matches} matches")

                        indices = indices.detach().cpu().numpy().reshape(-1, 2)
                        indices[:, 0] += offset1
                        indices[:, 1] += offset2
                        index_list.append(indices)

                if len(index_list) == 0:
                    continue

                indices = np.concatenate(index_list)
                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=indices)

    return

def merge_matches(
    feature_dir,
    input_matches_name_list,
    output_matches_name,
):
    with h5py.File(feature_dir / output_matches_name, mode="w") as f_matches:

        for input_matches_name in input_matches_name_list:
            with h5py.File(feature_dir / input_matches_name, mode="r") as f_input_matches:
                for key1 in f_input_matches.keys():
                    group  = f_matches.require_group(key1)
                    for key2 in f_input_matches[key1].keys():
                        group.create_dataset(key2, data=f_input_matches[key1][key2][...])

    return

# Match keypoints by LightGlue
##########################################

##########################################
# Check orientation
# https://github.com/ternaus/check_orientation

def convert_rot_k(index):
    if index == 0:
        return 0
    elif index == 1:
        return 3
    elif index == 2:
        return 2
    else:
        return 1

class CheckRotationDataset(Dataset):
    def __init__(self, files, transform=None):
        self.transform = transform
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imgPath = self.files[idx]
        image = T_read_image(str(imgPath), mode=ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        return image

def get_CheckRotation_dataloader(images, batch_size=1):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = CheckRotationDataset(images, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    return dataloader

def exec_rotation_detection(img_files):

    model = create_model("swsl_resnext50_32x4d")
    model.eval().cuda()

    dataloader = get_CheckRotation_dataloader(img_files)

    rots = []
    for idx, image in enumerate(dataloader):
        image = image.to(torch.float32).cuda()
        with torch.no_grad():
            prediction = model(image).detach().cpu().numpy()
            detected_rot = prediction[0].argmax()
            rot_k = convert_rot_k(detected_rot)
            rots.append(rot_k)
    return rots

def output_rot_images(
    paths,
    output_dir,
    rots,
):
    corrected_image_paths = []
    for rot, path in tqdm(zip(rots, paths), total=len(paths), desc=f"Rotating images", dynamic_ncols=True):

        img = cv2.imread(str(path))

        if rot == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rot == 3:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(str(output_dir / path.name), img)
        corrected_image_paths.append(output_dir / path.name)

    return corrected_image_paths

def exec_rotation_correction(paths, output_dir):

    rots = exec_rotation_detection(paths)

    corrected_image_paths = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            output_rot_images,
            np.array_split(paths, 2),
            itertools.repeat(output_dir),
            np.array_split(rots, 2),
        )
        for data in results:
            corrected_image_paths.append(data)

    corrected_image_paths = list(itertools.chain.from_iterable(corrected_image_paths))
    gc.collect()

    return corrected_image_paths

# Check orientation
##########################################

##########################################
# Doppelgangers: Learning to Disambiguate Images of Similar Structures
# https://github.com/RuojinCai/Doppelgangers

sys.path.append("/kaggle/input/doppelgangers-repo/doppelgangers")
from doppelgangers.third_party.loftr import LoFTR, default_cfg
from doppelgangers.utils.loftr_matches import read_image

def save_loftr_matches(data_path, pairs_info, pairs_info_index, output_path, device_id, model_weight_path="weights/outdoor_ds.ckpt"):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.

    device = torch.device(f"cuda:{device_id}")

    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_weight_path)['state_dict'])
    matcher = matcher.eval().to(device)

    img_size = 1024
    df = 8
    padding = True

    for _pairs_info, idx in tqdm(zip(pairs_info, pairs_info_index), total=len(pairs_info_index), desc=f"Running LOFTR [GPU{device_id}]"):

        output_file_path = output_path / "loftr_match" / f"{idx}.npy"
        if output_file_path.exists():
            continue

        name0, name1, _, _, _ = _pairs_info

        img0_pth = data_path / name0
        img1_pth = data_path / name1
        img0_raw, mask0 = read_image(str(img0_pth), img_size, df, padding)
        img1_raw, mask1 = read_image(str(img1_pth), img_size, df, padding)
        img0 = torch.from_numpy(img0_raw).to(device)
        img1 = torch.from_numpy(img1_raw).to(device)
        mask0 = torch.from_numpy(mask0).to(device)
        mask1 = torch.from_numpy(mask1).to(device)
        batch = {'image0': img0, 'image1': img1, 'mask0': mask0, 'mask1':mask1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            np.save(output_file_path, {"kpt0": mkpts0, "kpt1": mkpts1, "conf": mconf})

def doppelgangers_classifier(cfg, pretrained_model_path, pair_path):
    # basic setup
    cudnn.benchmark = True

    # initial dataset
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data)
    test_loader = loaders["test_loader"]

    # initial model
    decoder_lib = importlib.import_module(cfg.models.decoder.type)
    decoder = decoder_lib.decoder(cfg.models.decoder)
    decoder = decoder.cuda()

    # load pretrained model
    ckpt = torch.load(pretrained_model_path)
    new_ckpt = deepcopy(ckpt["dec"])
    for key, _ in ckpt["dec"].items():
        if "module." in key:
            new_ckpt[key[len("module."):]] = new_ckpt.pop(key)
    decoder.load_state_dict(new_ckpt, strict=True)

    # evaluate on test set
    decoder.eval()
    prob_list = list()
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Running doppelgangers"):

            # TTA
            img = data["image"].cuda()

            img_list = []
            img_list.append(img)
            # lrflip
            img = torch.flip(img, [3])
            img_list.append(img)
            # stack
            img = torch.cat(img_list)

            logits = decoder(img)
            prob = torch.nn.functional.softmax(logits, dim=1)
            prob = torch.mean(prob, dim=0)
            prob_list.append(prob[1].detach().cpu().numpy())

    y_scores = np.array(prob_list)

    pairs_info = np.load(pair_path, allow_pickle=True)
    pair_probability_file_path = cfg.data.output_path / "pair_probability_list.h5"
    with h5py.File(pair_probability_file_path, mode="w") as f_matches:

        for idx in range(pairs_info.shape[0]):

            key1, key2, _, _, _ = pairs_info[idx]
            score = y_scores[idx]

            group  = f_matches.require_group(key1)
            group.create_dataset(key2, data=score)

    return pair_probability_file_path

def exec_doppelgangers_classifier(
    images_dir,
    feature_dir,
    matches_path,
    num_gpus,
    loftr_weight_path,
    doppelgangers_weight_path,
):
    loftr_matches_path = feature_dir / "loftr_match"
    loftr_matches_path.mkdir(parents=True, exist_ok=True)

    def create_image_pair_list(matches_path, output_path):

        dummy = 0
        pairs_list = []
        with h5py.File(matches_path, mode="r") as f_matches:

            for key1 in f_matches.keys():
                group = f_matches[key1]
                for key2 in group.keys():
                    pairs_list.append([key1, key2, dummy, dummy, dummy])

        pairs_list = np.concatenate(pairs_list, axis=0).reshape(-1, 5)

        pairs_list_path = output_path / "pairs_list.npy"
        np.save(pairs_list_path, pairs_list)

        return pairs_list_path

    pair_path = create_image_pair_list(matches_path, feature_dir)
    pairs_info = np.load(pair_path, allow_pickle=True)
    pairs_info_index = np.arange(pairs_info.shape[0])

    with ThreadPoolExecutor() as executor:
        executor.map(
            save_loftr_matches,
            itertools.repeat(images_dir),
            np.array_split(pairs_info, num_gpus),
            np.array_split(pairs_info_index, num_gpus),
            itertools.repeat(feature_dir),
            range(num_gpus),
            itertools.repeat(loftr_weight_path),
        )
    gc.collect()
    torch.cuda.empty_cache()

    config = {
        "data": {
            "image_dir": images_dir,
            "loftr_match_dir": loftr_matches_path,
            "output_path": feature_dir,
            "type": "doppelgangers.datasets.sfm_disambiguation_dataset",
            "num_workers": 1,
            "test": {
                "batch_size": 1,
                "img_size": 1024,
                "pair_path": pair_path,
            },
        },
        "models": {
            "decoder": {
                "type": "doppelgangers.models.cnn_classifier",
                "input_dim": 10,
            },
        },
    }

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    config = dict2namespace(config)

    # Running Doppelgangers classifier model on image pairs
    print("Running Doppelgangers classifier model on image pairs")
    pair_probability_file_path = doppelgangers_classifier(config, doppelgangers_weight_path, pair_path)

    shutil.rmtree(loftr_matches_path)

    return pair_probability_file_path

# Doppelgangers: Learning to Disambiguate Images of Similar Structures
##########################################

##########################################
# Identify duplicate structures by 3D point cloud

sys.path.append("/kaggle/input/colmap-db-import")
from prepare_colmap import read_images_binary, read_cameras_binary, read_points3D_binary, project_3d_to_2d, get_camera_param

def get_points_inside_cameraview(
    T_pointcloud_camera, K, width, height, # target camera
    point_cloud, # 3D point cloud
):

    point_cloud_uv = project_3d_to_2d(T_pointcloud_camera, K, point_cloud)
    mask = (point_cloud_uv[:, 0] >= 0) & (point_cloud_uv[:, 0] < width) & (point_cloud_uv[:, 1] >= 0) & (point_cloud_uv[:, 1] < height)

    return point_cloud_uv[mask]

def keep_multi_camera_points(images, point_cloud_id, point_cloud, pcd_used_num_images_ratio):

    pcd_id_list = np.concatenate([np.unique(image['points_ids']) for image in images.values()])
    unique, counts = np.unique(pcd_id_list, return_counts=True)
    pcd_id_count = dict(zip(unique, counts))

    pcd_used_num_images = int(len(images) * pcd_used_num_images_ratio)
    filtered_pcd_ids = [k for k, v in pcd_id_count.items() if v > pcd_used_num_images]
    target_pcd_idx = np.isin(point_cloud_id, filtered_pcd_ids)

    return point_cloud_id[target_pcd_idx], point_cloud[:, target_pcd_idx]

def remove_ambiguous_area(paths, recon_data_dir, pcd_used_num_images_ratio, erode_mask_ratio):

    images = read_images_binary(recon_data_dir / "images.bin")
    cameras = read_cameras_binary(recon_data_dir / "cameras.bin")
    points = read_points3D_binary(recon_data_dir / "points3D.bin")

    point_cloud_id = points["id"].values
    point_cloud = points[['x', 'y', 'z']].values
    point_cloud = point_cloud.T

    # step1: Extract point cloud observed by more than half of the images
    point_cloud_id, point_cloud = keep_multi_camera_points(images, point_cloud_id, point_cloud, pcd_used_num_images_ratio)

    # step2: Cluster and remove outliers
    if len(point_cloud_id) < 10:
        num_labels = 0
    else:
        nearest_neighbors = NearestNeighbors(n_neighbors=10)
        neighbors = nearest_neighbors.fit(point_cloud.T)
        distances, _ = neighbors.kneighbors(point_cloud.T)
        distances = np.sort(distances[:,-1], axis=0)

        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        eps = distances[knee.knee]
        print(f"Knee: {knee.knee}, eps: {eps}")

        labels = DBSCAN(eps=eps, min_samples=5).fit(point_cloud.T).labels_
        num_labels = len(np.unique(labels))-1 if -1 in labels else len(np.unique(labels))

    print(f"Number of clusters: {num_labels}")

    kernel = np.ones((3,3), np.uint8)
    mask_dict = {}
    point_cloud_uv_dict = {}
    for path in tqdm(paths, total=len(paths), desc="remove ambiguous area"):

        key = path.name
        if key not in images:
            img = cv2.imread(str(path))
            height, width, _ = img.shape
            mask = np.zeros((height, width), dtype=np.uint8)
            mask_dict[key] = mask
            continue

        image = images[key]

        pcd_ids_cam = image['points_ids']
        T_pointcloud_camera, K, width, height = get_camera_param(cameras, image)

        all_mask = np.zeros((height, width), dtype=np.uint8)
        all_point_cloud_uv = []
        for label_id in range(num_labels):

            lbl_id = labels==label_id
            _point_cloud_id = point_cloud_id[lbl_id]
            _point_cloud = point_cloud[:,lbl_id]

            target_idx = np.isin(_point_cloud_id, pcd_ids_cam)
            point_cloud_cam = _point_cloud[:, target_idx]

            # step3: Get the point cloud in the camera view
            point_cloud_uv = get_points_inside_cameraview(T_pointcloud_camera, K, width, height, point_cloud_cam)

            # step4: Create a mask image
            mask = np.zeros((height, width), dtype=np.uint8)
            if len(point_cloud_uv) > 0:
                point_cloud_uv_int = point_cloud_uv.astype(int)
                mask[point_cloud_uv_int[:,1],point_cloud_uv_int[:,0]] = 255

                dilate_count = 0
                for i in range(1000):
                    n_labels, _ = cv2.connectedComponents(mask)
                    if n_labels <= 2:
                        break
                    mask = cv2.dilate(mask, kernel=kernel, iterations=1)
                    dilate_count += 1

                erode_count = int(dilate_count * erode_mask_ratio)
                if erode_count > 0:
                    mask = cv2.erode(mask, kernel=kernel, iterations=erode_count)

            all_mask[mask>0] = 255
            all_point_cloud_uv.append(point_cloud_uv)

        mask_dict[key] = all_mask
        point_cloud_uv_dict[key] = np.concatenate(all_point_cloud_uv, axis=0) if len(all_point_cloud_uv) > 0 else np.zeros((0, 2))

    return mask_dict, point_cloud_uv_dict

# Identify duplicate structures by 3D point cloud
##########################################

##########################################
# Prune the matching graphs caused by duplicate structures

def filter_FundamentalMatrix(mkpts0, mkpts1, filter_iterations=10, filter_threshold=8):

    store_inliers = { idx:0 for idx in range(mkpts0.shape[0]) }
    idxs = np.array(range(mkpts0.shape[0]))
    for _ in range(filter_iterations):
        try:
            Fm, inliers = cv2.findFundamentalMat(
                mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
            if Fm is not None:
                inliers = (inliers > 0).reshape(-1)
                inlier_idxs = idxs[inliers]
                for idx in inlier_idxs:
                    store_inliers[idx] += 1
        except:
            print(f"Failed to cv2.findFundamentalMat. mkpts.shape={mkpts0.shape}")
    inliers = np.array([ count for (idx, count) in store_inliers.items() ]) >= filter_threshold

    return inliers

def disambiguous_keypoints(mask_area, keypoints):
    disambiguous_idx = []
    for kpt in keypoints.astype(int):
        if kpt[0] < 0 or kpt[1] < 0 or kpt[0] >= mask_area.shape[1] or kpt[1] >= mask_area.shape[0]:
            disambiguous_idx.append(True)
            continue
        flag = (mask_area[kpt[1], kpt[0]] == 0)
        disambiguous_idx.append(flag)
    return np.array(disambiguous_idx)

def verify_matches(
    feature_dir,
    pair_probability_file_path,
    mask_area,
    doppelgangers_min_thr,
    doppelgangers_max_thr,
    filter_iterations,
    filter_threshold,
):

    index_pairs = []
    with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, \
         h5py.File(feature_dir / "offsets.h5", mode="r") as f_offsets_deg, \
         h5py.File(feature_dir / "matches.h5", mode="r") as f_matches, \
         h5py.File(pair_probability_file_path, mode="r") as f_scores:

        for key1 in tqdm(f_matches.keys(), desc="verify matches"):

            group = f_matches[key1]
            keypoints1 = f_keypoints[key1][...]
            mask_area_1 = pad_to_square(mask_area[key1])

            group_score = f_scores[key1]

            for key2 in group.keys():

                keypoints2 = f_keypoints[key2][...]
                matches = group[key2][...]
                mask_area_2 = pad_to_square(mask_area[key2])

                doppelgangers_score = group_score[key2][...]

                if doppelgangers_score < doppelgangers_min_thr:
                    continue

                if doppelgangers_score > doppelgangers_max_thr:
                    index_pairs.append((key1, key2, matches))
                    continue

                # rotate mask_area_2
                for deg in ["90deg", "180deg", "270deg"]:
                    offset = f_offsets_deg[key2][deg][...]
                    if offset > matches[0, 1]:
                        break
                    mask_area_2 = cv2.rotate(mask_area_2, cv2.ROTATE_90_CLOCKWISE)

                # Verify using Two-View Geometry
                mkpts1 = keypoints1[matches[:, 0]]
                mkpts2 = keypoints2[matches[:, 1]]
                inliers_fmat = filter_FundamentalMatrix(mkpts1, mkpts2, filter_iterations, filter_threshold)

                # Verify using Ambiguity Mask
                inliers_disambiguous_kpts1 = disambiguous_keypoints(mask_area_1, mkpts1)
                inliers_disambiguous_kpts2 = disambiguous_keypoints(mask_area_2, mkpts2)

                # Integrate verification results
                inliers = inliers_fmat & inliers_disambiguous_kpts1 & inliers_disambiguous_kpts2

                index_pairs.append((key1, key2, matches[inliers]))

    with h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:

        for key1, key2, matches in index_pairs:
            group  = f_matches.require_group(key1)
            group.create_dataset(key2, data=matches)

    return

# Prune the matching graphs caused by duplicate structures
##########################################

##########################################
# Get focal length from incremental mapping

def get_focal_length_prior(paths, recon_data_dir):

    images = read_images_binary(recon_data_dir / "images.bin")
    cameras = read_cameras_binary(recon_data_dir / "cameras.bin")

    focal_length_dict = {}
    for path in paths:

        key = path.name

        if key not in images:
            img = cv2.imread(str(path))
            height, width, _ = img.shape
            FOCAL_PRIOR = 1.2
            focal_length_dict[key] = FOCAL_PRIOR * max(height, width)
            continue

        _, K, _, _ = get_camera_param(cameras, images[key])

        focal_length_dict[key] = K[0][0]

    return focal_length_dict

# Get focal length from incremental mapping
##########################################

##########################################
# COLMAP Utils

sys.path.append("/kaggle/input/colmap-db-import")
from database import COLMAPDatabase
from h5_to_db import add_keypoints, add_matches

def import_into_colmap(
    path,
    feature_dir,
    database_path,
    camera_model,
    focal_length_dict=None,
    single_camera=False,
):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    fname_to_id = add_keypoints(
        db,
        feature_dir,
        path,
        camera_model,
        single_camera,
        focal_length_dict,
    )
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()

    return

# COLMAP Utils
##########################################

##########################################
# Submission Utils

def parse_sample_submission(data_type, scene_type):

    df_scene = pd.read_csv(f"/kaggle/input/image-matching-challenge-2024/{data_type}/categories.csv")
    if data_type == "train":
        df = pd.read_csv("/kaggle/input/imc2024-validation/validation.csv")
        if scene_type != "submission":
            df = df[df["scene"] == scene_type]
            df_scene = df_scene[df_scene["scene"] == scene_type]
    else:
        df = pd.read_csv("/kaggle/input/image-matching-challenge-2024/sample_submission.csv")

    data_dict = {}
    category_dict = {}
    for dataset in df["dataset"].unique():
        data_dict[dataset] = {}
        category_dict[dataset] = {}
        for scene in df[df["dataset"] == dataset]["scene"].unique():
            data_dict[dataset][scene] = []

            image_path_list = df[(df["dataset"] == dataset) & (df["scene"] == scene)]["image_path"].tolist()
            for image_path in image_path_list:
                data_dict[dataset][scene].append(Path(f"/kaggle/input/image-matching-challenge-2024/{image_path}"))

            category_dict[dataset][scene] = df_scene[df_scene["scene"] == scene]["categories"].values[0]

    return data_dict, category_dict

def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])

def create_submission(
    output_dir,
    results,
    data_dict,
    base_path,
    scene_type,
):
    """Prepares a submission file."""

    with open(output_dir / f"{scene_type}.csv", "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")

        for dataset in data_dict:
            # Only write results for datasets with images that have results
            if dataset in results:
                res = results[dataset]
            else:
                res = {}

            # Same for scenes
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R":{}, "t":{}}

                # Write the row with rotation and translation matrices
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print(image)
                        R = scene_res[image]["R"].reshape(-1)
                        T = scene_res[image]["t"].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    image_path = str(image.relative_to(base_path))
                    f.write(f"{image_path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")
    return

# Submission Utils
##########################################

############################
# Feature Matching

def feature_matching_default(
    config,
    feature_dir,
    image_paths,
    index_pairs,
    trial_index,
):
    # Detect keypoints of all images
    with ThreadPoolExecutor() as executor:
        executor.map(
            detect_keypoints,
            np.array_split(image_paths, config.num_gpus),
            itertools.repeat(feature_dir),
            range(config.num_gpus),
            itertools.repeat(config.keypoint_detection_args["num_features"]),
            itertools.repeat(config.keypoint_detection_args["detection_threshold"]),
            itertools.repeat(config.keypoint_detection_args["resize_to"]),
        )
    torch.cuda.empty_cache()
    gc.collect()

    merge_single_h5(
        feature_dir,
        [ f"keypoints{i}.h5" for i in range(config.num_gpus) ],
        "keypoints.h5",
    )
    for i in range(config.num_gpus):
        (feature_dir / f"keypoints{i}.h5").unlink()

    for file_prefix in ["descriptors_deg", "keypoints_deg", "offsets"]:
        merge_double_h5(
            feature_dir,
            [ f"{file_prefix}{i}.h5" for i in range(config.num_gpus) ],
            f"{file_prefix}.h5",
        )
        for i in range(config.num_gpus):
            (feature_dir / f"{file_prefix}{i}.h5").unlink()

    # Match keypoints of pairs of similar images
    with ThreadPoolExecutor() as executor:
        executor.map(
            keypoint_distances,
            itertools.repeat(image_paths),
            np.array_split(index_pairs, config.num_gpus),
            itertools.repeat(feature_dir),
            range(config.num_gpus),
            itertools.repeat(config.keypoint_distances_args["early_stopping_thr"]),
            itertools.repeat(config.keypoint_distances_args["min_matches"][trial_index]),
            itertools.repeat(config.keypoint_distances_args["verbose"]),
        )
    (feature_dir / "descriptors_deg.h5").unlink()
    (feature_dir / "keypoints_deg.h5").unlink()
    torch.cuda.empty_cache()
    gc.collect()

    # Merge the matching results of each GPU
    merge_matches(
        feature_dir,
        [ f"matches{i}.h5" for i in range(config.num_gpus) ],
        "matches.h5",
    )
    for i in range(config.num_gpus):
        (feature_dir / f"matches{i}.h5").unlink()
    gc.collect()

    return

# Special processing for transparent objects
def feature_matching_transparent(
    config,
    feature_dir,
    image_paths,
    index_pairs,
    trial_index,
):
    # Correct the orientation of the images
    corrected_images_dir = feature_dir / "corrected_images"
    corrected_images_dir.mkdir(parents=True, exist_ok=True)
    corrected_image_paths = exec_rotation_correction(
        image_paths,
        corrected_images_dir,
    )
    torch.cuda.empty_cache()
    gc.collect()

    # Extract foreground masks of all images
    with ThreadPoolExecutor() as executor:
        executor.map(
            dinov2_segmentation,
            np.array_split(corrected_image_paths, config.num_gpus),
            itertools.repeat(feature_dir),
            range(config.num_gpus),
        )
    torch.cuda.empty_cache()
    gc.collect()

    merge_single_h5(
        feature_dir,
        [ f"fg_mask{i}.h5" for i in range(config.num_gpus) ],
        "fg_mask.h5",
    )
    for i in range(config.num_gpus):
        (feature_dir / f"fg_mask{i}.h5").unlink()
    gc.collect()

    # Detect keypoints of all images
    num_grids = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            detect_keypoints_transparent,
            np.array_split(corrected_image_paths, config.num_gpus),
            itertools.repeat(feature_dir),
            range(config.num_gpus),
            itertools.repeat(config.keypoint_detection_transparent_args["num_features"]),
            itertools.repeat(config.keypoint_detection_transparent_args["detection_threshold"]),
            itertools.repeat(config.keypoint_detection_transparent_args["resize_to"]),
        )
        for data in results:
            num_grids.append(data)
    num_grids = max(num_grids)
    torch.cuda.empty_cache()
    gc.collect()

    (feature_dir / "fg_mask.h5").unlink()
    shutil.rmtree(corrected_images_dir)

    merge_single_h5(
        feature_dir,
        [ f"keypoints{i}.h5" for i in range(config.num_gpus) ],
        "keypoints.h5",
    )
    for i in range(config.num_gpus):
        (feature_dir / f"keypoints{i}.h5").unlink()

    for file_prefix in ["descriptors_grid", "keypoints_grid", "offsets_grid"]:
        merge_double_h5(
            feature_dir,
            [ f"{file_prefix}{i}.h5" for i in range(config.num_gpus) ],
            f"{file_prefix}.h5",
        )
        for i in range(config.num_gpus):
            (feature_dir / f"{file_prefix}{i}.h5").unlink()

    gc.collect()

    # Match keypoints of pairs of similar images
    with ThreadPoolExecutor() as executor:
        executor.map(
            keypoint_distances_transparent,
            itertools.repeat(corrected_image_paths),
            np.array_split(index_pairs, config.num_gpus),
            itertools.repeat(feature_dir),
            itertools.repeat(num_grids),
            range(config.num_gpus),
            itertools.repeat(config.keypoint_distances_args["min_matches"][trial_index]),
            itertools.repeat(config.keypoint_distances_args["verbose"]),
        )
    (feature_dir / "descriptors_grid.h5").unlink()
    (feature_dir / "keypoints_grid.h5").unlink()
    (feature_dir / "offsets_grid.h5").unlink()
    torch.cuda.empty_cache()
    gc.collect()

    # Merge the matching results of each GPU
    merge_matches(
        feature_dir,
        [ f"matches{i}.h5" for i in range(config.num_gpus) ],
        "matches.h5",
    )
    for i in range(config.num_gpus):
        (feature_dir / f"matches{i}.h5").unlink()
    gc.collect()

    return

def feature_matching(args, config):

    trial_index = args["trial_index"]
    image_paths = args["image_paths"]
    categories = args["categories"]
    feature_dir = args["feature_dir"]

    if feature_dir.exists():
        shutil.rmtree(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    database_path = feature_dir / "database.db"

    # Get the pairs of images that are somewhat similar
    index_pairs = get_pairs_exhaustive(image_paths)
    gc.collect()

    # KeyPoint Detection and Matching
    if "transparent" in categories:
        feature_matching_transparent(
            config,
            feature_dir,
            image_paths,
            index_pairs,
            trial_index,
        )
    else:
        feature_matching_default(
            config,
            feature_dir,
            image_paths,
            index_pairs,
            trial_index,
        )

    # Import keypoint distances of matches into colmap
    import_into_colmap(
        image_paths[0].parent,
        feature_dir,
        database_path,
        camera_model = "simple-radial" if ("historical_preservation" in categories) or ("transparent" in categories) else "simple-pinhole",
        single_camera = True if "transparent" in categories else False,
    )
    if ("symmetries-and-repeats" not in categories):
        for file_path in feature_dir.glob("*.h5"):
            file_path.unlink()
    if ("transparent" in categories):
        for file_path in feature_dir.glob("*.h5"):
            file_path.unlink()

    results = {
        "image_paths": image_paths,
        "categories": categories,
        "feature_dir": feature_dir,
        "database_path": database_path,
    }
    gc.collect()

    return results

# Feature Matching
############################

############################
# 3D Reconstruction

def reconstruction(args, config):

    feature_dir = args["feature_dir"]
    image_paths = args["image_paths"]
    database_path = args["database_path"]

    # 4.2. Feature Matching and Geometric Verification
    pycolmap.match_exhaustive(database_path)
    gc.collect()

    # 5.1 Incrementally start reconstructing the scene (sparse reconstruction)
    # The process starts from a random pair of images and is incrementally extended by
    # registering new images and triangulating new points.
    output_path = feature_dir / "colmap_rec_aliked"
    output_path.mkdir(parents=True, exist_ok=True)

    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_paths[0].parent,
        output_path=output_path,
        options=pycolmap.IncrementalPipelineOptions(**config.colmap_mapper_options),
    )
    gc.collect()

    # 5.2. Look for the best reconstruction: The incremental mapping offered by
    # pycolmap attempts to reconstruct multiple models, we must pick the best one
    images_registered  = 0
    num_points3D = 0
    best_idx = None

    print ("Looking for the best reconstruction")

    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            try:
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    num_points3D = len(rec.points3D)
                    best_idx = idx1
                elif len(rec.images) == images_registered:
                    if len(rec.points3D) > num_points3D:
                        images_registered = len(rec.images)
                        num_points3D = len(rec.points3D)
                        best_idx = idx1

            except Exception:
                continue

    if best_idx is not None:
        results = {}
        results["images_registered"] = images_registered
        results["num_points3D"] = num_points3D
        results["best_idx"] = best_idx
        results["maps"] = deepcopy(maps)
    else:
        results = None
    gc.collect()

    return results

def refine_symmetries_and_repeats(
    config,
    feature_dir,
    image_paths,
    recon_data_dir,
):
    # 6.1 Correct the orientation of the images
    corrected_images_dir = feature_dir / "corrected_images"
    corrected_images_dir.mkdir(parents=True, exist_ok=True)
    exec_rotation_correction(image_paths, corrected_images_dir)
    torch.cuda.empty_cache()
    gc.collect()

    # 6.2 Classify doppelgangers
    pair_probability_file_path = exec_doppelgangers_classifier(
        corrected_images_dir,
        feature_dir,
        feature_dir / "matches.h5",
        config.num_gpus,
        **config.doppelgangers_classifier_args,
    )
    shutil.rmtree(corrected_images_dir)
    torch.cuda.empty_cache()
    gc.collect()

    # 6.3 Remove duplicate structures
    mask_area, _ = remove_ambiguous_area(
        image_paths,
        recon_data_dir,
        **config.remove_ambiguous_area_args,
    )
    gc.collect()

    # 6.4 Prune the matching graphs caused by duplicate structures
    verify_matches(
        feature_dir,
        pair_probability_file_path,
        mask_area,
        **config.verify_matches_args,
    )
    for file_path in feature_dir.glob("*.npy"):
        file_path.unlink()
    gc.collect()

    # 7. Reconstruct the scene again
    database_path = feature_dir / "database_refine.db"
    if database_path.exists():
        database_path.unlink()

    import_into_colmap(
        image_paths[0].parent,
        feature_dir,
        database_path,
        camera_model = "simple-pinhole", # TODO:"simple-radial"
        focal_length_dict = get_focal_length_prior(image_paths, recon_data_dir),
        single_camera = False,
    )
    for file_path in feature_dir.glob("*.h5"):
        file_path.unlink()

    pycolmap.match_exhaustive(database_path)
    gc.collect()

    output_path = feature_dir / "colmap_rec_aliked_refine"
    output_path.mkdir(parents=True, exist_ok=True)

    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_paths[0].parent,
        output_path=output_path,
        options=pycolmap.IncrementalPipelineOptions(**config.colmap_mapper_options),
    )
    gc.collect()

    images_registered  = 0
    best_idx = None

    print ("Looking for the best reconstruction")

    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            try:
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx1
            except Exception:
                continue

    if not DEBUG:
        shutil.rmtree(feature_dir)

    if best_idx is not None:
        results = {}
        results["best_idx"] = best_idx
        results["maps"] = deepcopy(maps)
    else:
        results = None
    gc.collect()

    return results

# 3D Reconstruction
############################

############################
# Pipeline process

def prepare_input(data_dict, config):

    datasets = list(data_dict.keys())

    image_num_list = []
    dataset_list = []
    scene_list = []
    trial_list = []
    for dataset in datasets:
        for scene in data_dict[dataset]:
            image_num = len(data_dict[dataset][scene])
            for trial_idx in range(config.num_trials):
                dataset_list.append(dataset)
                scene_list.append(scene)
                trial_list.append(trial_idx)
                image_num_list.append(image_num)

    # sort by the number of images
    dataset_list = np.array(dataset_list)
    scene_list = np.array(scene_list)
    trial_list = np.array(trial_list)
    image_num_list = np.array(image_num_list)
    sort_idx = np.argsort(image_num_list)[::-1]
    dataset_list = dataset_list[sort_idx]
    scene_list = scene_list[sort_idx]
    trial_list = trial_list[sort_idx]

    return dataset_list, scene_list, trial_list

# core function for multithreading
import threading
import queue

def wrap_func_for_mt(func, params):
    def wrap_func(queue_input, queue_output):
        while True:
            input = queue_input.get()
            if input is None:
                queue_output.put(None)
                continue

            result = func(input, params)

            queue_output.put(result)

    return wrap_func

def loop_proc(queues_input, queues_output, inputs):
    for queue_input, input in zip(queues_input, inputs):
        queue_input.put(input)

    outputs = []
    for queue_output in queues_output:
        output = queue_output.get()
        outputs.append(output)

    return outputs

def prepare_multithreading(params):

    # funcs to proc in pipeline
    func_params = [
        (feature_matching, (params)),
        (reconstruction, (params)),
    ]
    wrap_funcs = list(map(lambda func_param: wrap_func_for_mt(func_param[0], func_param[1]), func_params))

    # prepare queues
    queues_input = [queue.Queue() for _ in range(len(wrap_funcs))]
    queues_output = [queue.Queue() for _ in range(len(wrap_funcs))]

    # create Threads
    threads = []
    for wrap_func, queue_input, queue_output in zip(wrap_funcs, queues_input, queues_output):
        t = threading.Thread(target=wrap_func, args=(queue_input, queue_output), daemon=True)
        threads.append(t)

    for t in threads:
        t.start()

    return queues_input, queues_output, len(wrap_funcs)

# Pipeline process
############################

##########################################
# Main process

def get_pairs_exhaustive(lst):
    return list(itertools.combinations(range(len(lst)), 2))

def run_from_config(config):

    data_dict, category_dict = parse_sample_submission(DATA_TYPE, SCENE_TYPE)
    datasets = list(data_dict.keys())

    ############################
    # Pipeline process
    dataset_list, scene_list, trial_list = prepare_input(data_dict, config)

    queues_input, queues_output, len_wrap_funcs = prepare_multithreading(config)

    idx = 0
    results_trial_list = []
    image_paths_list = {}
    while len(results_trial_list) < len(dataset_list):

        if idx >= len(dataset_list):
            args = None
        else:
            dataset = dataset_list[idx]
            scene = scene_list[idx]
            trial_idx = trial_list[idx]

            # use all images in the directory
            images_dir = data_dict[dataset][scene][0].parent
            image_paths = list(images_dir.glob("*"))

            if len(image_paths) > 100:
                # random sampling
                test_image_paths = data_dict[dataset][scene]
                additional_image_paths = [path for path in image_paths if path not in test_image_paths]

                random.shuffle(additional_image_paths)
                additional_image_size = 100 - len(test_image_paths)
                additional_image_paths = additional_image_paths[:additional_image_size]

                image_paths = test_image_paths + additional_image_paths

            if dataset not in image_paths_list:
                image_paths_list[dataset] = {}

            if scene not in image_paths_list[dataset]:
                image_paths_list[dataset][scene] = {}

            image_paths_list[dataset][scene][trial_idx] = image_paths

            args = {
                "trial_index": trial_idx,
                "image_paths": image_paths,
                "categories": category_dict[dataset][scene],
                "feature_dir": config.feature_dir / f"{dataset}_{scene}_trial{trial_idx}",
            }

        if idx == 0:
            init_inputs = [args] + [None]*(len_wrap_funcs - 1)  # [[], None, None, ...]
            inputs = init_inputs
        else:
            inputs = [args] + outputs[:-1]

        outputs = loop_proc(queues_input, queues_output, inputs)
        result = outputs[-1]

        if result is not None:
            results_trial_list.append(result)

        idx = idx + 1

    ############################
    # Pipeline post-processing
    results_trial = {}
    for dataset, scene, trial_idx, results in zip(dataset_list, scene_list, trial_list, results_trial_list):

        if dataset not in results_trial:
            results_trial[dataset] = {}

        if scene not in results_trial[dataset]:
            results_trial[dataset][scene] = {}

        results_trial[dataset][scene][trial_idx] = results

    ############################
    # Gather the best results with the most images registered and the most 3D points
    for dataset in datasets:
        for scene in data_dict[dataset]:

            best_trial_ID = 0
            images_registered_trial = 0
            num_points3D_trial = 0
            maps_trial = None
            best_idx_trial = None
            for trial_idx in results_trial[dataset][scene].keys():

                recon_result = results_trial[dataset][scene][trial_idx]

                if recon_result is not None:

                    flag = False
                    if recon_result["images_registered"] > images_registered_trial:
                        flag = True
                    elif recon_result["images_registered"] == images_registered_trial:
                        if recon_result["num_points3D"] > num_points3D_trial:
                            flag = True

                    if flag:
                        best_trial_ID = trial_idx
                        images_registered_trial = recon_result["images_registered"]
                        num_points3D_trial = recon_result["num_points3D"]
                        best_idx_trial = recon_result["best_idx"]
                        maps_trial = deepcopy(recon_result["maps"])

            if best_idx_trial is not None:
                print(f"Best Trial is {best_trial_ID}")
                print(f"Best idx:{best_idx_trial}, num_images:{images_registered_trial}, num_points3D:{num_points3D_trial}")

                results_trial[dataset][scene]["best_trial_ID"] = best_trial_ID
                results_trial[dataset][scene]["best_idx"] = best_idx_trial
                results_trial[dataset][scene]["maps"] = deepcopy(maps_trial)

    ##########################
    # Parse the reconstruction object to get the rotation matrix and translation vector
    # symmetries-and-repeats: refine the reconstruction
    results = {}
    for dataset in datasets:

        if dataset not in results:
            results[dataset] = {}

        for scene in data_dict[dataset]:

            best_idx = None
            categories = category_dict[dataset][scene]

            if ("symmetries-and-repeats" in categories) and ("transparent" not in categories):

                best_trial_ID = results_trial[dataset][scene]["best_trial_ID"]
                best_idx = results_trial[dataset][scene]["best_idx"]

                feature_dir = config.feature_dir / f"{dataset}_{scene}_trial{best_trial_ID}"
                recon_data_dir = feature_dir / "colmap_rec_aliked" / f"{best_idx}"
                image_paths = image_paths_list[dataset][scene][best_trial_ID]

                _result = refine_symmetries_and_repeats(
                    config,
                    feature_dir,
                    image_paths,
                    recon_data_dir,
                )

                if _result is not None:
                    maps = _result["maps"]
                    best_idx = _result["best_idx"]

            else:
                maps = results_trial[dataset][scene]["maps"]
                best_idx = results_trial[dataset][scene]["best_idx"]

            # Parse the reconstruction object to get the rotation matrix and translation vector
            # obtained for each image in the reconstruction
            results[dataset][scene] = {}
            if best_idx is not None:
                for k, im in maps[best_idx].images.items():
                    key = config.base_path / f"{DATA_TYPE}" / scene / "images" / im.name
                    results[dataset][scene][key] = {}
                    results[dataset][scene][key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                    results[dataset][scene][key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

            gc.collect()


    if not DEBUG:
        for dataset in datasets:
            for scene in data_dict[dataset]:
                for trial_idx in range(config.num_trials):
                    feature_dir = config.feature_dir / f"{dataset}_{scene}_trial{trial_idx}"
                    if feature_dir.exists():
                        shutil.rmtree(feature_dir)

    create_submission(output_dir, results, data_dict, config.base_path, SCENE_TYPE)

# config
class Config:
    base_path = Path("/kaggle/input/image-matching-challenge-2024")
    feature_dir = output_dir / ".feature_outputs"

    device = K.utils.get_cuda_device_if_available(0)
    num_gpus = torch.cuda.device_count()

    num_trials = 2

    keypoint_detection_args = {
        "num_features": 4096,
        "detection_threshold": 0.01,
        "resize_to": 1024,
    }

    keypoint_detection_transparent_args = {
        "num_features": 4096,
        "detection_threshold": 0.5,
        "resize_to": 1024,
    }

    keypoint_distances_args = {
        "early_stopping_thr": 1000,
        "min_matches": [100, 125],
        "verbose": False,
    }

    doppelgangers_classifier_args = {
        "loftr_weight_path": Path("/kaggle/input/loftr-pytorch-outdoor-v1/loftr_outdoor.ckpt"),
        "doppelgangers_weight_path": Path("/kaggle/input/doppelgangers-repo/doppelgangers/weights/doppelgangers_classifier_loftr.pt"),
    }

    remove_ambiguous_area_args = {
        "pcd_used_num_images_ratio": 0.5,
        "erode_mask_ratio": 0.0,
    }

    verify_matches_args = {
        "doppelgangers_min_thr": 0.3,
        "doppelgangers_max_thr": 0.5,
        "filter_iterations": 1,
        "filter_threshold": 1,
    }

    colmap_mapper_options = {
        "min_model_size": 3,
        "max_num_models": 2,
    }

run_from_config(Config)