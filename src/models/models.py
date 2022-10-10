#!/usr/bin/env python
import tensorflow as tf
import torch
import torchvision

import transformers

from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from src.models.weenet import DenseWeeNet, Conv2dWeeNet
from src.models.gans.dcgan import Generator as DCGAN_Generator
from src.models.gans.dcgan import Discriminator as DCGAN_Discriminator
from src.models.gans.BigGAN.BigGAN import Generator as BigGAN_Generator
import numpy as np

# we need lazy loading of models to avoid memory issues


def load_keras_transformer_model(module, name, seq_len, batch_size):
    model = module.from_pretrained(name)
    dummy_input = tf.keras.Input(shape=[seq_len], batch_size=batch_size, dtype="int32")
    dummy_out = model(dummy_input)  # Propagate shapes through the keras model.
    np_input = np.random.uniform(
        size=[batch_size, seq_len], low=0, high=seq_len
    ).astype("int32")
    outs = model(np_input).logits
    np_out = outs.numpy()
    return model, np_input, np_out


def convert_to_hf_transformer_graphdef(model, batch_size, seq_len):
    model_func = tf.function(lambda x: model(x))
    input_dict = model._saved_model_inputs_spec
    input_spec = input_dict[list(input_dict.keys())[0]]
    model_func = model_func.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len], input_spec.dtype)
    )
    frozen_func = convert_variables_to_constants_v2(model_func)
    return frozen_func.graph.as_graph_def()


def get_hf_tf_model(name, batch_size, seq_len):
    if "google/mobilebert-uncased" == name:
        module = getattr(transformers, "TFMobileBertForSequenceClassification")
    elif "bert-base-uncased" == name:
        module = getattr(transformers, "TFBertForSequenceClassification")
    elif "gpt2" in name:
        module = getattr(transformers, "TFGPT2ForSequenceClassification")
    else:
        raise ValueError("Unknown architecture", name)

    model, dummy_input, dummy_out = load_keras_transformer_model(
        module, name=name, batch_size=batch_size, seq_len=seq_len
    )

    return (
        convert_to_hf_transformer_graphdef(model, batch_size, seq_len),
        dummy_input,
        dummy_out,
    )


def get_bert_hf_trans(batch_size, seq_len):
    model_name = "bert-base-uncased"
    model, dummy_input, dummy_out = get_hf_tf_model(model_name, batch_size, seq_len)

    test_inputs = {"input_1": (dummy_input, "int32")}

    return model, test_inputs, dummy_out


def get_mobilebert_hf_trans(batch_size, seq_len):
    model_name = "google/mobilebert-uncased"
    model, dummy_input, dummy_out = get_hf_tf_model(model_name, batch_size, seq_len)

    test_inputs = {"input_1": (dummy_input, "int32")}

    return model, test_inputs, dummy_out


# def get_gpt2_hf_trans(batch_size, seq_len):
#     model_name = "gpt2"
#     model, dummy_input, dummy_out = get_hf_tf_model(model_name, batch_size, seq_len)

#     test_inputs = {"input_1": (dummy_input, "int32")}

#     return model, test_inputs, dummy_out


def get_DCGAN_Generator_pyt():
    # default values
    nz, ngf, nc = 100, 64, 3

    model = DCGAN_Generator(0, nz, ngf, nc)
    return model


def get_DCGAN_Discriminator_pyt():
    # default values
    ndf, nc = 64, 3

    model = DCGAN_Discriminator(0, nc, ndf)
    return model


def get_BigGAN_Generator_pyt():
    model = BigGAN_Generator()
    return model


def get_dense_weenet_1():
    model = DenseWeeNet(256, 1000)
    return model


def get_dense_weenet_2():
    model = DenseWeeNet(4096, 1000)
    return model


def get_dense_weenet_3():
    model = DenseWeeNet(512, 256)
    return model


def get_conv2d_weenet_1():
    model = Conv2dWeeNet(64, 128)
    return model


def get_conv2d_weenet_2():
    model = Conv2dWeeNet(128, 512)
    return model


def get_conv2d_weenet_3():
    model = Conv2dWeeNet(256, 256)
    return model


def get_stylegan2_G():
    from src.models.gans.stylegan2_ada_pytorch.training.networks import Generator
    import requests
    import tempfile
    import pickle

    url = (
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl"
    )
    req = requests.get(url)

    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
        print(temp_file.name)
        with open(temp_file.name, "wb") as f:
            f.write(req.content)

        with open(temp_file.name, "rb") as f:
            G = pickle.load(f)["G_ema"].cpu()  # torch.nn.Module

    # create a new network using the new defintion
    G2 = Generator(
        G.z_dim,  # Input latent (Z) dimensionality.
        G.c_dim,  # Conditioning label (C) dimensionality.
        G.w_dim,  # Intermediate latent (W) dimensionality.
        G.img_resolution,  # Output resolution.
        G.img_channels,
    ).cpu()

    # update the weights to match your trained model
    g_sd = G.state_dict()
    g2_sd = G2.state_dict()

    for k, _ in g2_sd.items():
        g2_sd[k] = g_sd[k]

    G2.load_state_dict(g2_sd)

    import functools

    G2.forward = functools.partial(G2.forward, c=None, force_fp32=True)

    return G2
