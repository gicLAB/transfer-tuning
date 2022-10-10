#!/usr/bin/env python
# import tensorflow as tf
import torch
import torchvision


from src.models.weenet import WeeNet
from src.models.models import *

# we need lazy loading of models to avoid memory issues
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

chocolate = {
    # CNNs
    "resnet18": torchvision.models.resnet18,
    "resnet50": torchvision.models.resnet50,
    "alexnet": torchvision.models.alexnet,
    "vgg16": torchvision.models.vgg16,
    "mobilenetv2": torchvision.models.mobilenet_v2,
    "efficentnetb0": torchvision.models.efficientnet_b0,
    "efficentnetb4": torchvision.models.efficientnet_b4,
    "googlenet": torchvision.models.googlenet,
    "mnasnet1_0": torchvision.models.mnasnet1_0,
    # transformers
    "bert-base-uncased-seq_class-128": get_bert_hf_trans,
    "bert-base-uncased-seq_class-256": get_bert_hf_trans,
    "mobilebert-base-uncased-seq_class-128": get_mobilebert_hf_trans,
    "mobilebert-base-uncased-seq_class-256": get_mobilebert_hf_trans,

}

model_sets = {
    # "vanilla": vanilla,
    "chocolate": chocolate,
    # "test": test,
    # "dense_vs_conv2d": dense_vs_conv2d,
}

# defines the dataset/input data shape that a model is defined for
# in the case where in the input shape can vary, we choose a common value
dataset_dict = {
    "mobilenetv2": "imagenet",
    "resnet50": "imagenet",
    "resnet18": "imagenet",
    "googlenet": "imagenet",
    "efficentnetb0": "imagenet",
    "efficentnetb4": "imagenet",
    "vgg16": "imagenet",
    "alexnet": "imagenet",
    "mnasnet1_0": "imagenet",
    "bert-base-uncased-seq_class-128": "seq_class-128",
    "bert-base-uncased-seq_class-256": "seq_class-256",
    "bert-base-uncased-seq_class-768": "seq_class-768",
    "mobilebert-base-uncased-seq_class-128": "seq_class-128",
    "mobilebert-base-uncased-seq_class-256": "seq_class-256",
    "mobilebert-base-uncased-seq_class-768": "seq_class-768",
}

# defines the framework which a given model is defined in
# key for exporting it correctly
framework_dict = {
    "mobilenetv1": "keras",
    "mobilenetv2": "pytorch",
    "mobilenetv3": "pytorch",
    "resnet50": "pytorch",
    "resnet18": "pytorch",
    "efficentnetb0": "pytorch",
    "efficentnetb4": "pytorch",
    "efficentnetb7": "pytorch",
    "squeezenet1_0": "pytorch",
    "vgg16": "pytorch",
    "alexnet": "pytorch",
    "shufflenet_v2_x1_0": "pytorch",
    "inceptionv3": "pytorch",
    "densenet161": "pytorch",
    "densenet201": "pytorch",
    "googlenet": "pytorch",
    "mnasnet0_5": "pytorch",
    "mnasnet1_0": "pytorch",
    "weenet": "pytorch",
    "bert-base-uncased-seq_class-128": "hf-transformers",
    "bert-base-uncased-seq_class-256": "hf-transformers",
    "bert-base-uncased-seq_class-768": "hf-transformers",
    "mobilebert-base-uncased-seq_class-128": "hf-transformers",
    "mobilebert-base-uncased-seq_class-256": "hf-transformers",
    "mobilebert-base-uncased-seq_class-768": "hf-transformers",
    "gpt2-seq_class-256": "hf-transformers",
    "dcgan_generator_nz_100_ngf_64": "pytorch",
    "dcgan_discriminator_nz_100_ndf_64": "pytorch",
    "biggan_generator_b1_z128": "pytorch",
    "dense_weenet1": "pytorch",
    "dense_weenet2": "pytorch",
    "dense_weenet3": "pytorch",
    "conv2d_weenet1": "pytorch",
    "conv2d_weenet2": "pytorch",
    "conv2d_weenet3": "pytorch",
    "stylegan2_G_afhqdog": "pytorch",
}

data_shape_dict = {
    "imagenet": [((1, 3, 224, 224), "float32")],
    "weenet1": [((1, 3, 224, 224), "float32")],
    "seq_class-128": (1, 128),
    "seq_class-256": (1, 256),
    "seq_class-768": (1, 768),
    "dcgan_noise_100": [((1, 100, 1, 1), "float32")],
    "dcgan_discriminator_input_3_64": [((1, 3, 64, 64), "float32")],
    "biggan_noise_b1_z128": [
        ([1, 128], "float32"),
        (
            [
                1,
            ],
            "float32",
        ),
    ],  # z: [batch, dim_z], y: [batch]
    "dense_weenet1": [((1, 256), "float32")],
    "dense_weenet2": [((1, 4096), "float32")],
    "dense_weenet3": [((1, 512), "float32")],
    "conv2d_weenet1": [((1, 64, 128, 128), "float32")],
    "conv2d_weenet2": [((1, 128, 64, 64), "float32")],
    "conv2d_weenet3": [((1, 256, 64, 64), "float32")],
    "stylegan2_G_afhqdog": [((1, 512), "float32")],
}

out_shape_dict = {"imagenet": (1000,)}


def get_model_set_data(model_set):
    in_shapes = dict()
    frameworks = dict()
    for name, model in model_set.items():
        dataset = dataset_dict[name]
        framework = framework_dict[name]
        in_shape = data_shape_dict[dataset]

        if not isinstance(in_shape, list):
            in_shape = [in_shape]
        in_shapes[name] = in_shape
        frameworks[name] = framework
    return in_shapes, frameworks
