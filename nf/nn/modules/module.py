import numpy as np
from collections import OrderedDict
import itertools
from nf import Tensor

__all__ = ["Module"]


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module(object):
    """
    Module模块大部分参照torch的Module模块，因为nf设定与torch模型兼容，所以此处代码可直接从torch抄来啦。
    已完成一般参数从torch模型中导入。
    """
    _version = 1.0
    def __init__(self):
        self.trainable = True
        self._buffers = OrderedDict()
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *input):
        raise NotImplementedError

    def _get_name(self):
        return self.__class__.__name__

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def register_parameter(self, name, param):
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError("属性已存在 '{}'".format(name))
        if not isinstance(param, Tensor):
            raise TypeError("无法将 '{}' 类型设为 '{}' "
                            "(权值参数需要 nf.Tensor 类型)"
                            .format(type(param), name))
        elif not param.requires_grad:
            raise ValueError(
                "权值 {} 梯度应当开启".format(name))
        self._parameters[name] = param

    def __getattr__(self, name):
        """
        结合 __setattr__ 中的 remove_from ，可以完成属性的隐藏和可控地访问。
        :param name:
        :return:
        """
        _parameters = self.__dict__['_parameters']
        if name in _parameters:
            return _parameters[name]
        _modules = self.__dict__['_modules']
        if name in _modules:
            return _modules[name]
        _buffers = self.__dict__['_buffers']
        if name in _buffers:
            return _buffers[name]
        raise AttributeError("'{}' 属性不存在".format(name))

    def __setattr__(self, name, value):
        """
        从 __dict__ 中删除某个属性，是为了让外部可控地访问该属性，相当于java中可控的private
        NOTE: modules中所有可求梯度的tensor一律视为权值，所以对于那些非权值的Tensor属性，将其requires_grad设为False！
        :param name:
        :param value:
        :return:
        """
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if isinstance(value, Tensor) and value.requires_grad:   # modules中所有可求梯度的tensor一律视为权值
            remove_from(self.__dict__, self._modules, self._buffers)
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            remove_from(self.__dict__, self._parameters, self._buffers)
            self._modules[name] = value
        elif isinstance(value, Tensor) and not value.requires_grad:
            remove_from(self.__dict__, self._parameters, self._modules)
            self._buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        state_dict = itertools.chain(self._parameters.items(), self._buffers.items())
        for name, param in state_dict:
            if param is not None:
                destination[prefix + name] = param if keep_vars else param


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        return destination

    def _load_from_state_dict(self, state_dict, prefix,
                              missing_keys, unexpected_keys, error_msgs):
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    if input_param.size == param.size:
                        input_param = input_param.reshape(param.shape)
                    else:
                        # local shape should match the one in checkpoint
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                          'the shape in current model is {}.'
                                          .format(key, input_param.shape, param.shape))
                        continue

                if isinstance(input_param, np.ndarray):
                    # backwards compatibility for serialized parameters
                    input_param = Tensor(input_param)
                try:
                    # print("ppp",id(param), param.requires_grad)
                    param.copy_(input_param)
                    # print("ppp", id(param), param.requires_grad)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.shape, input_param.shape))
            else:
                missing_keys.append(key)

        for key in state_dict.keys():
            if key.startswith(prefix):
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)

    def load_state_dict(self, state_dict):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        def load(module, prefix=''):
            module._load_from_state_dict(state_dict, prefix, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        load = None  # break load->load reference cycle

        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))

    def __call__(self, *inputs):
        return self.forward(*inputs)


