
import argparse
from data.sampler import arguments_sampler
from options.utils import load_config_file
from data.datasets import arguments_dataset
from cvnets import arguments_model, arguments_nn_layers, arguments_ema
from loss_fn import arguments_loss_fn
from optim import arguments_optimizer
from optim.scheduler import arguments_scheduler
from metrics import arguments_stats
from data.transforms import arguments_augmentation
from typing import Optional


def arguments_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="Common arguments", description="Common arguments")

    group.add_argument('--common.seed', type=int, default=0, help='Random seed')
    group.add_argument('--common.config-file', type=str,
                       default='config/detection/edgeformer/ssd_edgeformer_s.yaml')
    group.add_argument('--common.results-loc', type=str, default="/home/disk/result/classificaion_imagenet1k", help="Directory where results will be stored")
    group.add_argument('--common.run-label', type=str, default='run_1', help='Label id for the current run')

    group.add_argument('--common.resume', type=str, default=None, help='Resume location')
    group.add_argument('--common.finetune', type=str, default=None, help='Checkpoint location to be used for finetuning')
    group.add_argument('--common.finetune-ema', type=str, default=None,
                       help='EMA Checkpoint location to be used for finetuning')

    group.add_argument('--common.mixed-precision', action='store_true', help='Mixed precision training')
    group.add_argument('--common.accum-freq', type=int, default=1,
                       help='Accumulate gradients for this number of iterations')
    group.add_argument('--common.accum-after-epoch', type=int, default=0,
                       help='Start accumulation after this many epochs')
    group.add_argument('--common.log-freq', type=int, default=100, help='Display after these many iterations')
    group.add_argument('--common.auto-resume', action="store_true", help='Resume training from the last checkpoint')
    group.add_argument('--common.grad-clip', type=float, default=None, help="Gradient clipping value")
    group.add_argument('--common.k-best-checkpoints', type=int, default=5, help="Keep k-best checkpoints")
    return parser


def arguments_ddp(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="DDP arguments", description="DDP arguments")
    group.add_argument('--ddp.enable', action="store_true", help="Use DDP")
    group.add_argument('--ddp.rank', type=int, default=0, help="Node rank for distributed training")
    group.add_argument('--ddp.world-size', type=int, default=1, help="World size for DDP")
    group.add_argument('--ddp.dist-url', type=str, default=None, help="DDP URL")
    group.add_argument('--ddp.dist-port', type=int, default=6006, help="DDP Port")
    return parser


def get_training_arguments(parse_args: Optional[bool] = True):
    parser = argparse.ArgumentParser(description='Training arguments', add_help=True)

    # sampler related arguments
    parser = arguments_sampler(parser=parser)

    # dataset related arguments
    parser = arguments_dataset(parser=parser)

    # transform related arguments
    parser = arguments_augmentation(parser=parser)

    # model related arguments
    parser = arguments_nn_layers(parser=parser)
    parser = arguments_model(parser=parser)
    parser = arguments_ema(parser=parser)

    # loss function arguments
    parser = arguments_loss_fn(parser=parser)

    # optimizer arguments
    parser = arguments_optimizer(parser=parser)
    parser = arguments_scheduler(parser=parser)

    ## DDP arguments
    parser = arguments_ddp(parser=parser)

    ## stats arguments
    parser = arguments_stats(parser=parser)

    ## common
    parser = arguments_common(parser=parser)

    if parse_args:
        # parse args
        opts = parser.parse_args()
        opts = load_config_file(opts)
        return opts
    else:
        return parser


def get_eval_arguments():
    return get_training_arguments(parse_args=True)


def get_conversion_arguments():
    parser = get_training_arguments(parse_args=False)

    #
    group = parser.add_argument_group('Conversion/Espressobar arguments')
    group.add_argument('--conversion.coreml-extn', type=str, default="mlmodel", help="Extension for converted model. Default is mlmodel")

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts


def get_segmentation_eval_arguments():
    parser = get_training_arguments(parse_args=False)

    group = parser.add_argument_group("Segmentation evaluation related arguments")
    group.add_argument("--evaluation.segmentation.apply-color-map", action="store_true",
                       help="Apply color map to different classes in segmentation masks. Useful in visualization "
                            "+ some competitions (e.g, PASCAL VOC) accept submissions with colored segmentation masks")
    group.add_argument("--evaluation.segmentation.save-overlay-rgb-pred", action="store_true",
                       help="enable this flag to visualize predicted masks on top of input image")
    group.add_argument("--evaluation.segmentation.save-masks", action="store_true",
                       help="save predicted masks without colormaps. Useful for submitting to "
                            "competitions like Cityscapes")
    group.add_argument("--evaluation.segmentation.overlay-mask-weight", default=0.5,
                       help="Contribution of mask when overlaying on top of RGB image. ")
    group.add_argument("--evaluation.segmentation.mode", type=str, default="single_image", required=True,
                       choices=["single_image", "image_folder", "validation_set"],
                       help="Contribution of mask when overlaying on top of RGB image. ")
    group.add_argument("--evaluation.segmentation.path", type=str, default=None,
                       help="Path of the image or image folder (only required for single_image and image_folder modes)")
    group.add_argument("--evaluation.segmentation.num-classes", type=str, default=None,
                       help="Number of segmentation classes used during training")
    group.add_argument("--evaluation.segmentation.resize-input-images", action="store_true",
                       help="Resize input images")

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts


def get_detection_eval_arguments():
    parser = get_training_arguments(parse_args=False)

    group = parser.add_argument_group("Detection evaluation related arguments")
    group.add_argument("--evaluation.detection.save-overlay-boxes", action="store_true",
                       help="enable this flag to visualize predicted masks on top of input image")
    group.add_argument("--evaluation.detection.mode", type=str, default="validation_set", required=True,
                       choices=["single_image", "image_folder", "validation_set"],
                       help="Contribution of mask when overlaying on top of RGB image. ")
    group.add_argument("--evaluation.detection.path", type=str, default=None,
                       help="Path of the image or image folder (only required for single_image and image_folder modes)")
    group.add_argument("--evaluation.detection.num-classes", type=str, default=None,
                       help="Number of segmentation classes used during training")
    group.add_argument("--evaluation.detection.resize-input-images", action="store_true", default=False,
                       help="Resize the input images")

    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts


def get_tracking_eval_arguments():
    parser = get_training_arguments(parse_args=False)

    group = parser.add_argument_group("Tracking evaluation related arguments")
    group.add_argument("--evaluation.detection.save-overlay-boxes", action="store_true",
                       help="enable this flag to visualize predicted masks on top of input image")
    group.add_argument("--evaluation.detection.mode", type=str, default="validation_set", required=True,
                       choices=["single_image", "image_folder", "validation_set"],
                       help="Contribution of mask when overlaying on top of RGB image. ")
    group.add_argument("--evaluation.detection.path", type=str, default=None,
                       help="Path of the image or image folder (only required for single_image and image_folder modes)")
    group.add_argument("--evaluation.detection.num-classes", type=str, default=None,
                       help="Number of segmentation classes used during training")
    group.add_argument("--evaluation.detection.resize-input-images", action="store_true", default=False,
                       help="Resize the input images")
    parser.add_argument("demo", default="image", help="demo type")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    parser.add_argument("--output_dir", type=str, default="output") 
    parser.add_argument("--test_size", type=tuple, default=(320, 320))
    
    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)

    return opts