#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import resq.config as config
from resq.util import Singleton


class Traffic(object, metaclass=Singleton):
    def __init__(self, name):
        self.name = name
        self.desc = None
        self.dist_spec = None
        self.nr_flows = None
        self.size = None

        self.__dict__.update(config.traffics[name])

    def __str__(self):
        return 'Traffic(%s)' % (self.name)


class TrafficManager(object, metaclass=Singleton):
    def __init__(self):
        self._store = {}
        for name in config.traffics.keys():
            self.register(Traffic(name))

    def list(self, retname=False):
        return self._store.keys() if retname else self._store.values()

    def list_names(self, *args, **kwargs):
        return self.list(*args, **kwargs, retname=True)

    def register(self, traffic):
        self._store[traffic.name] = traffic
