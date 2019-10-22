#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
import zerorpc

c = zerorpc.Client(timeout=100)
c.connect("tcp://0.0.0.0:8688")
print(c.generate('黄种人', '微笑程度', [-1., 0., 1.]))
