#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
import zerorpc

c = zerorpc.Client()
c.connect("tcp://127.0.0.1:8688")
print c.listinfo("this is test string")
print c.generate('黄种人', '微笑程度', [-1., 0., 1.])
