#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>

models = {'黄种人': 'generator_yellow.pkl', '网红脸': 'generator_wanghong.pkl', '动漫老婆': 'generator_dongman.pkl'}

latents = {'微笑程度': 'latent_directions/smile.npy',
           '生气程度': 'latent_directions/emotion_angry.npy',
           '开心程度': 'latent_directions/emotion_happy.npy',
           '人脸左右角度': 'latent_directions/angle_horizontal.npy',
           '人脸上下角度': 'latent_directions/angle_vertical.npy',
           '颜值': 'latent_directions/beauty.npy'}
