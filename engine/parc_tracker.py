from curses import raw
import os
import os.path as osp
import time
import multiprocessing
import copy
from typing import Optional

import cv2
import torch
from torch.cuda.amp import autocast
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np

from options.opts import get_tracking_eval_arguments
from cvnets import get_model
from cvnets.models.detection.ssd import DetectionPredTuple
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master
from utils.tensor_utils import to_numpy, tensor_size_from_opts
from utils.color_map import Colormap
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from utils import logger
from engine.utils import print_summary


FONT_SIZE = cv2.FONT_HERSHEY_PLAIN
LABEL_COLOR = [255, 255, 255]
TEXT_THICKNESS = 1
RECT_BORDER_THICKNESS = 2
COLOR_MAP = Colormap().get_box_color_codes()


def predict(opts,
            input_tensor: Tensor,
            model: nn.Module,
            input_arr: Optional[np.ndarray] = None,
            device: Optional = torch.device("cpu"),
            mixed_precision_training: Optional[bool] = False,
            is_validation: Optional[bool] = False,
            file_name: Optional[str] = None,
            output_stride: Optional[int] = 32, # Default is 32 because ImageNet models have 5 downsampling stages (2^5 = 32)
            orig_h: Optional[int] = None,
            orig_w: Optional[int] = None,
            timer: Timer = None
            ):

    if input_arr is None and not is_validation:
        input_arr = (
            to_numpy(input_tensor) # convert to numpy
            .squeeze(0) # remove batch dimension
        )

    curr_height, curr_width = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_height // output_stride) * output_stride
    new_w = (curr_width // output_stride) * output_stride

    if new_h != curr_height or new_w != curr_width:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(input=input_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # move data to device
    input_tensor = input_tensor.to(device)

    with autocast(enabled=mixed_precision_training):
        # prediction
        timer.tic()
        prediction: DetectionPredTuple = model.predict(input_tensor, is_scaling=False)

    # convert tensors to boxes
    boxes = prediction.boxes.cpu().numpy()
    labels = prediction.labels.cpu().numpy()
    scores = prediction.scores.cpu().numpy()

    if orig_w is None:
        assert orig_h is None
        orig_h, orig_w = input_arr.shape[:2]
    elif orig_h is None:
        assert orig_w is None
        orig_h, orig_w = input_arr.shape[:2]

    assert orig_h is not None and orig_w is not None
    boxes[..., 0::2] = boxes[..., 0::2] * orig_w
    boxes[..., 1::2] = boxes[..., 1::2] * orig_h
    boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2])
    boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2])

    # raw_boxes = copy.deepcopy(boxes)
    # boxes[..., 0] = raw_boxes[..., 0] - raw_boxes[..., 2] / 2
    # boxes[..., 1] = raw_boxes[..., 1] - raw_boxes[..., 3] / 2
    # boxes[..., 2] = raw_boxes[..., 0] + raw_boxes[..., 2] / 2
    # boxes[..., 3] = raw_boxes[..., 1] + raw_boxes[..., 3] / 2

    return boxes, labels, scores




def predict_imageflow(opts, **kwargs):
    device = getattr(opts, "dev.device", torch.device('cpu'))
    mixed_precision_training = getattr(opts, "common.mixed_precision", False)
    # set-up the model
    model = get_model(opts)
    model.eval()
    model = model.to(device=device)
    print_summary(opts=opts, model=model)

    if model.training:
        logger.warning('Model is in training mode. Switching to evaluation mode')
        model.eval()
    
    output_dir = osp.join(opts.output_dir, "parc_tracker")
    os.makedirs(output_dir, exist_ok=True)

    if opts.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
        
    if opts.trt:
        opts.device = "gpu"
    opts.device = torch.device("cuda" if opts.device == "gpu" else "cpu")

    logger.info("Args: {}".format(opts))


    with torch.no_grad():
        current_time = time.localtime()
        imageflow_demo(model, vis_folder, current_time, opts)
        

def inference(model, img, timer, opts):
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = osp.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None
    
    device = getattr(opts, "dev.device", torch.device('cpu'))
    mixed_precision_training = getattr(opts, "common.mixed_precision", False)

    img_copy = copy.deepcopy(img)
    orig_h, orig_w = img_copy.shape[:2]
    img_info["height"] = orig_h
    img_info["width"] = orig_w
    img_info["raw_img"] = img
    
    # Resize the image to the resolution that detector supports
    res_h, res_w = tensor_size_from_opts(opts)
    
    img = np.array(img)
    ratio = min(res_h / img.shape[0], res_w / img.shape[1])
    img_info["ratio"] = ratio
    
    
    input_img = cv2.resize(img, (res_h, res_w), interpolation=cv2.INTER_LINEAR)

    # HWC --> CHW
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = (
        torch.div(
            torch.from_numpy(input_img).float(),  # convert to float tensor
            255.0  # convert from [0, 255] to [0, 1]
        ).unsqueeze(dim=0)  # add a dummy batch dimension
    )
    boxes, labels, scores = predict(
                                opts=opts,
                                input_tensor=input_img,
                                input_arr=img_copy,
                                model=model,
                                device=device,
                                mixed_precision_training=mixed_precision_training,
                                is_validation=False,
                                orig_h=orig_h,
                                orig_w=orig_w,
                                timer=timer
                            )
    filters = (labels == 1)
    new_boxes = boxes[filters]
    new_labels = labels[filters]
    new_scores = scores[filters]
    new_scores = np.expand_dims(new_scores, axis=0)
    new_labels = np.expand_dims(new_labels, axis=0)
    outputs = np.concatenate((new_boxes, new_scores.T), 1)
    return outputs, img_info


def imageflow_demo(detector, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = inference(detector, frame, timer, args)
            if outputs is not None:
                online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")



def main_track_demo(**kwargs):
    opts = get_tracking_eval_arguments()
    
    dataset_name = getattr(opts, "dataset.name", "imagenet")
    if dataset_name.find("coco") > -1:
        # replace model specific datasets (e.g., coco_ssd) with general COCO dataset
        setattr(opts, "dataset.name", "coco")
    
    opts = device_setup(opts)
    
    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error('--rank should be >=0. Got {}'.format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = '{}/{}'.format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    logger.log("Results (if any) will be stored here: {}".format(exp_dir))

    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    if num_gpus < 2:
        cls_norm_type = getattr(opts, "model.normalization.name", "batch_norm_2d")
        if cls_norm_type.find("sync") > -1:
            # replace sync_batch_norm with standard batch norm on PU
            setattr(opts, "model.normalization.name", cls_norm_type.replace("sync_", ""))
            setattr(opts, "model.classification.normalization.name", cls_norm_type.replace("sync_", ""))

    # we disable the DDP setting for evaluation tasks
    setattr(opts, "ddp.use_distributed", False)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    if dataset_workers == -1:
        setattr(opts, "dataset.workers", n_cpus)

    # We are not performing any operation like resizing and cropping on images
    # Because image dimensions are different, we process 1 sample at a time.
    setattr(opts, "dataset.train_batch_size0", 1)
    setattr(opts, "dataset.val_batch_size0", 1)
    setattr(opts, "dev.device_id", None)

    num_seg_classes = getattr(opts, "model.detection.n_classes", 81)
    assert num_seg_classes is not None
    
    predict_imageflow(opts=opts, **kwargs)



if __name__ == "__main__":
    main_track_demo()