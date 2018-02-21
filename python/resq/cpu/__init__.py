#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from resq.util import Singleton
from resq.util import list_cores


class Core(object, metaclass=Singleton):
    def __init__(self, name):
        self.name = name
        with open('/sys/devices/system/cpu/cpu%d/topology/physical_package_id'
                  % name, 'r') as f:
            self.numa_node = int(f.read())

    def __str__(self):
        return 'Core(%s)' % (self.name)

    @property
    def is_available(self):
        return CoreManager().is_available(self)

    def free(self):
        return CoreManager().free(self)

    def reserve(self):
        return CoreManager().reserve(self)


class CoreManager(object, metaclass=Singleton):
    def __init__(self):
        self._store = {}
        self._reserved = set()
        for name in list_cores():
            self.register(Core(name))

    def is_available(self, core):
        return core not in self._reserved

    def free(self, core):
        if core not in self._reserved:
            raise ValueError('Core %d is already free' % core.name)
        self._reserved.remove(core)

    def list(self,
             available=None,
             numa_node=None,
             retname=False):
        return [k if retname else v for k, v in self._store.items()
                if (available is None or v.is_available == available) and
                (numa_node is None or v.numa_node == numa_node)]

    def list_names(self, *args, **kwargs):
        return self.list(*args, **kwargs, retname=True)

    def reserve(self, core):
        if core in self._reserved:
            raise ValueError('Core %d is already reserved' % core.name)
        self._reserved.add(core)

    def register(self, core):
        self._store[core.name] = core
