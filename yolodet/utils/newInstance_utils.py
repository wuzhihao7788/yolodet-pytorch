#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''=================================================
# @Time    : 2020/7/27 10:43
# @Author  : wuzhihao
# @email   : 753993117@qq.com
# @FileName: newInstance_utils.py
# @Software: PyCharm
# @github ：https://github.com/wuzhihao7788/yolodet-pytorch

               ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃              ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━-┓
                ┃Beast god bless┣┓
                ┃　Never BUG ！ ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
=================================================='''
import sys

def createInstance(module_name, class_name, *args, **kwargs):
    module_meta = __import__(module_name, globals(), locals(), [class_name])
    class_meta = getattr(module_meta, class_name)
    obj = class_meta(*args, **kwargs)
    return obj

def build_from_dict(cfg,registry,default_args=None):
    assert isinstance(cfg, dict) and 'type' in cfg
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type,str):
        clazz = registry[obj_type]
        clazz_name = clazz.split('.')[-1]
        module_name = clazz[0:len(clazz) - len(clazz_name) - 1]
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        obj = createInstance(module_name=module_name, class_name=clazz_name, **args)
        return obj

    return None

def obj_from_dict(info, parent=None, default_args=None):
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type,str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)
