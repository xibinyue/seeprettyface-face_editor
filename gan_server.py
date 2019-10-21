#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
import zerorpc
import pickle
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
from util.generator_model import Generator
import os

from play_with_dlatent import read_feature, generate_image


class ImageGenerator(object):
    def __init__(self):
        tflib.init_tf()
        with open('model/generator_yellow.pkl', "rb") as f:
            self.generator_network, self.discriminator_network, self.Gs_network = pickle.load(f)
        self.generator = Generator(self.Gs_network, batch_size=1, randomize_noise=False)

    def move_latent_and_save(self, latent_vector, direction, coeffs, generator):
        res = []
        for i, coeff in enumerate(coeffs):
            new_latent_vector = latent_vector.copy()
            new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
            result = generate_image(new_latent_vector, generator)
            dst_path = os.path.join('results', '%s.png' % str(i).zfill(3))
            result.save(dst_path)
            res.append(result)
        return res

    def generate(self):
        face_latent = read_feature('input_latent/0001.txt')
        stack_latents = np.stack(face_latent for _ in range(1))
        face_dlatent = self.Gs_network.components.mapping.run(stack_latents, None)
        direction = np.load('latent_directions/smile.npy')
        coeffs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4.]
        imgs_list = self.move_latent_and_save(face_dlatent, direction, coeffs, self.generator)
        return imgs_list


def server_gan():
    image_gen = ImageGenerator()
    s = zerorpc.Server(image_gen)
    print('build server...')
    s.bind("tcp://0.0.0.0:8688")
    print('bind port...')
    s.run()
    print('server runing....')

if __name__ == '__main__':
    server_gan()
