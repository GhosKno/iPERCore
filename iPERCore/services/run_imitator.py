# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import glob
import numpy as np

from iPERCore.models import ModelsFactory
from iPERCore.services.preprocess import preprocess
from iPERCore.services.personalization import personalize
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.base_runner import add_hands_params_to_smpl
from iPERCore.services.options.meta_info import MetaOutput

from iPERCore.tools.utils.signals.smooth import temporal_smooth_smpls
from iPERCore.tools.utils.filesio.cv_utils import load_parse, read_cv2_img, normalize_img
from iPERCore.tools.utils.filesio.persistence import clear_dir
from iPERCore.tools.utils.multimedia.video import fuse_source_reference_output, fuse_src_ref_multi_outputs

from scipy.spatial.transform import Rotation as R


def get_src_info_for_inference(opt, vid_info):
    img_dir = vid_info["img_dir"]
    src_ids = vid_info["src_ids"]
    image_names = vid_info["images"]

    alpha_paths = vid_info["alpha_paths"]
    inpainted_paths = vid_info["inpainted_paths"]
    actual_bg_path = vid_info["actual_bg_path"]

    masks = []
    for i in src_ids:
        parse_path = alpha_paths[i]
        mask = load_parse(parse_path, opt.image_size)
        masks.append(mask)

    if actual_bg_path is not None:
        bg_img = read_cv2_img(actual_bg_path)
        bg_img = normalize_img(bg_img, image_size=opt.image_size, transpose=True)

    elif opt.use_inpaintor:
        bg_img = read_cv2_img(inpainted_paths[0])
        bg_img = normalize_img(bg_img, image_size=opt.image_size, transpose=True)

    else:
        bg_img = None

    src_info_for_inference = {
        "paths": [os.path.join(img_dir, image_names[i]) for i in src_ids],
        "smpls": vid_info["smpls"][src_ids],
        "offsets": vid_info["offsets"],
        "links": vid_info["links"],
        "masks": masks,
        "bg": bg_img
    }

    return src_info_for_inference

def random_affine_smpls(smpls, scale):
    # smpls: n*85 [:, 3,72,10]
    n = smpls.shape[0]
    random_pose = np.random.uniform(-scale, scale, (n, 72))
    smpls[:, 3:-10] += random_pose
    return smpls

def random_affine_smpls_with_weights(smpls, w1=0.05, w2=0.15, w3=0.3):
    inds_l1 = (np.array([0,1,2,3,6,9]).repeat(3).reshape([-1, 3]) * 3 + np.array([0,1,2])).reshape(-1)
    inds_l2 = (np.array([4,5,12,13,14,15,16,17]).repeat(3).reshape([-1, 3]) * 3 + np.array([0, 1, 2])).reshape(-1)

    random_pose = np.random.uniform(-w3, w3, smpls[:, 3:-10].shape)
    random_pose[:, inds_l1] = np.random.uniform(-w1, w1, random_pose[:, inds_l1].shape)
    random_pose[:, inds_l2] = np.random.uniform(-w2, w2, random_pose[:, inds_l2].shape)
    smpls[:, 3:-10] += random_pose
    return smpls

def add_view_effect(smpls, view_dir):
    """

    Args:
        smpls (np.ndarray): (n, 85)
        view_dir (float):

    Returns:
        smpls (np.ndarray): (n, 85)
    """
    length = len(smpls)

    r = R.from_euler("xyz", [0, view_dir, 0], degrees=True)

    for i in range(length):
        orig_r = R.from_rotvec(smpls[i, 3:6])
        cur_r = (r * orig_r).as_rotvec()
        smpls[i, 3:6] = cur_r

    return smpls


def imitate(opt):
    """

    Args:
        opt:

    Returns:
        all_meta_outputs (list of MetaOutput):

    """

    print("Step 3: running imitator.")

    if opt.ip:
        from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
        visualizer = VisdomVisualizer(env=opt.model_id, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    # set imitator
    imitator = ModelsFactory.get_by_name("imitator", opt)

    meta_src_proc = opt.meta_data["meta_src"]
    meta_ref_proc = opt.meta_data["meta_ref"]

    all_meta_outputs = []
    for i, meta_src in enumerate(meta_src_proc):
        """
        meta_input:
                path: /p300/tpami/neuralAvatar/sources/fange_1/fange_1_ns=2
                bg_path: /p300/tpami/neuralAvatar/sources/fange_1/IMG_7225.JPG
                name: fange_1
        primitives_dir: ../tests/debug/primitives/fange_1
        processed_dir: ../tests/debug/primitives/fange_1/processed
        vid_info_path: ../tests/debug/primitives/fange_1/processed/vid_info.pkl
        """
        src_proc_info = ProcessInfo(meta_src)
        src_proc_info.deserialize()

        src_info = src_proc_info.convert_to_src_info(num_source=opt.num_source)
        src_info_for_inference = get_src_info_for_inference(opt, src_info)

        # 1. personalization
        imitator.source_setup(
            src_path=src_info_for_inference["paths"],
            src_smpl=src_info_for_inference["smpls"],
            masks=src_info_for_inference["masks"],
            bg_img=src_info_for_inference["bg"],
            offsets=src_info_for_inference["offsets"],
            links_ids=src_info_for_inference["links"],
            visualizer=visualizer
        )

        for j, meta_ref in enumerate(meta_ref_proc):
            """
            meta_input:
                path: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp4
                bg_path:
                name: bantangzhuyi_1
                audio: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp3
                fps: 30.02
                pose_fc: 400.0
                cam_fc: 150.0
            primitives_dir: ../tests/debug/primitives/bantangzhuyi_1
            processed_dir: ../tests/debug/primitives/bantangzhuyi_1/processed
            vid_info_path: ../tests/debug/primitives/bantangzhuyi_1/processed/vid_info.pkl
            """

            ref_proc_info = ProcessInfo(meta_ref)
            ref_proc_info.deserialize()

            ref_info = ref_proc_info.convert_to_ref_info()
            ref_imgs_paths = ref_info["images"]
            ref_smpls = ref_info["smpls"]
            # ref_smpls = random_affine_smpls(ref_smpls, opt.random_scale)
            ref_smpls = add_hands_params_to_smpl(ref_smpls, imitator.body_rec.np_hands_mean)

            meta_output = MetaOutput(meta_src, meta_ref)

            # if there are more than 10 frames, then we will use temporal smooth of smpl.
            if len(ref_smpls) > 10:
                ref_smpls = temporal_smooth_smpls(ref_smpls, pose_fc=meta_output.pose_fc, cam_fc=meta_output.cam_fc)

            out_imgs_dir = clear_dir(meta_output.imitation_dir)

            outputs = imitator.inference(tgt_paths=ref_imgs_paths, tgt_smpls=ref_smpls,
                                         cam_strategy=opt.cam_strategy, output_dir=out_imgs_dir,
                                         visualizer=visualizer, verbose=True)

            fuse_source_reference_output(
                meta_output.imitation_mp4, src_info_for_inference["paths"],
                ref_imgs_paths,
                outputs,
                # sorted(glob.glob(os.path.join(meta_output.imitation_dir, "pred_*"))),
                audio_path=meta_output.audio, fps=meta_output.fps,
                image_size=opt.image_size, pool_size=opt.num_workers
            )

            all_meta_outputs.append(meta_output)

    for meta_output in all_meta_outputs:
        print(meta_output)

    print("Step 3: running imitator done.")
    return all_meta_outputs

def imitate_with_multi_random_pose(opt):
    """

    Args:
        opt:

    Returns:
        all_meta_outputs (list of MetaOutput):

    """

    print("Step 3: running imitator.")

    if opt.ip:
        from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
        visualizer = VisdomVisualizer(env=opt.model_id, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    # set imitator
    imitator = ModelsFactory.get_by_name("imitator", opt)

    meta_src_proc = opt.meta_data["meta_src"]
    meta_ref_proc = opt.meta_data["meta_ref"]

    all_meta_outputs = []
    for i, meta_src in enumerate(meta_src_proc):
        """
        meta_input:
                path: /p300/tpami/neuralAvatar/sources/fange_1/fange_1_ns=2
                bg_path: /p300/tpami/neuralAvatar/sources/fange_1/IMG_7225.JPG
                name: fange_1
        primitives_dir: ../tests/debug/primitives/fange_1
        processed_dir: ../tests/debug/primitives/fange_1/processed
        vid_info_path: ../tests/debug/primitives/fange_1/processed/vid_info.pkl
        """
        src_proc_info = ProcessInfo(meta_src)
        src_proc_info.deserialize()

        src_info = src_proc_info.convert_to_src_info(num_source=opt.num_source)
        src_info_for_inference = get_src_info_for_inference(opt, src_info)

        # 1. personalization
        imitator.source_setup(
            src_path=src_info_for_inference["paths"],
            src_smpl=src_info_for_inference["smpls"],
            masks=src_info_for_inference["masks"],
            bg_img=src_info_for_inference["bg"],
            offsets=src_info_for_inference["offsets"],
            links_ids=src_info_for_inference["links"],
            visualizer=visualizer
        )

        for j, meta_ref in enumerate(meta_ref_proc):
            """
            meta_input:
                path: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp4
                bg_path:
                name: bantangzhuyi_1
                audio: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp3
                fps: 30.02
                pose_fc: 400.0
                cam_fc: 150.0
            primitives_dir: ../tests/debug/primitives/bantangzhuyi_1
            processed_dir: ../tests/debug/primitives/bantangzhuyi_1/processed
            vid_info_path: ../tests/debug/primitives/bantangzhuyi_1/processed/vid_info.pkl
            """

            ref_proc_info = ProcessInfo(meta_ref)
            ref_proc_info.deserialize()

            ref_info = ref_proc_info.convert_to_ref_info()
            ref_imgs_paths = ref_info["images"]
            ref_smpls = ref_info["smpls"]
            multi_out_img_paths = []
            for k in range(6):
                if k == 0:
                    ref_smpls_tmp = ref_smpls
                elif k <= 2:
                    ref_smpls_tmp = add_view_effect(ref_smpls.copy(), -10 if k == 1 else 10)
                else:
                    ref_smpls_tmp = random_affine_smpls_with_weights(ref_smpls.copy())
                ref_smpls_tmp = add_hands_params_to_smpl(ref_smpls_tmp, imitator.body_rec.np_hands_mean)

                meta_output = MetaOutput(meta_src, meta_ref)

                # if there are more than 10 frames, then we will use temporal smooth of smpl.
                if len(ref_smpls_tmp) > 10:
                    ref_smpls_tmp = temporal_smooth_smpls(ref_smpls_tmp, pose_fc=meta_output.pose_fc, cam_fc=meta_output.cam_fc)

                out_imgs_dir = clear_dir(meta_output.imitation_dir+'/{}'.format(str(k)))

                outputs = imitator.inference(tgt_paths=ref_imgs_paths, tgt_smpls=ref_smpls_tmp,
                                             cam_strategy=opt.cam_strategy, output_dir=out_imgs_dir,
                                             visualizer=visualizer, verbose=True)
                multi_out_img_paths.append(outputs)
            multi_out_img_paths = [[o[i] for o in multi_out_img_paths] for i in range(len(multi_out_img_paths[0]))]
            fuse_src_ref_multi_outputs(
                meta_output.imitation_mp4, src_info_for_inference["paths"],
                ref_imgs_paths,
                multi_out_img_paths,
                # sorted(glob.glob(os.path.join(meta_output.imitation_dir, "pred_*"))),
                fps=meta_output.fps,
                image_size=opt.image_size, pool_size=opt.num_workers
            )

            all_meta_outputs.append(meta_output)

    for meta_output in all_meta_outputs:
        print(meta_output)

    print("Step 3: running imitator done.")
    return all_meta_outputs

def run_imitator(opt):
    # 1. prepreocess
    successful = preprocess(opt)

    if successful:
        # 2. personalization
        personalize(opt)
        # 3. imitate
        # all_meta_outputs = imitate(opt)
        all_meta_outputs = imitate_with_multi_random_pose(opt)
    else:
        all_meta_outputs = []

    return all_meta_outputs


if __name__ == "__main__":
    from iPERCore.services.options.options_inference import InferenceOptions

    OPT = InferenceOptions().parse()
    run_imitator(opt=OPT)

