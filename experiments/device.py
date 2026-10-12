"""Device visibility must be restricted before importing JAX."""

import os


def configure_device(args):
    if args.gpu is not None and (args.platform != "cuda" or args.gpu not in range(4)):
        raise ValueError("gpu must be physical index 0,1,2,3 and requires platform=cuda")
    os.environ["JAX_PLATFORMS"] = args.platform
    if args.platform == "cuda":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0 if args.gpu is None else args.gpu)
