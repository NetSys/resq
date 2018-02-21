#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
import os
from string import Template
import sys
import pytoml as toml


def __expand_vars(d):
    for k, v in d.items():
        if isinstance(v, dict):
            __expand_vars(v)
        if isinstance(v, str):
            d[k] = Template(v).safe_substitute(os.environ)


def __load():
    global nfs, perf_events, ports, traffics
    for k, v in os.environ.items():
        if k in ['HANDLE', 'PORTS', 'CORES'] or \
                k.startswith('PORT') or \
                k.startswith('PCIADDR') or \
                k.startswith('CORE'):
            del os.environ[k]
    mod = sys.modules[__name__]
    cfg_dir = os.environ.get('RESQ_CFG_PATH')
    if not cfg_dir:
        print('Please source the ResQ environment file', file=sys.stderr)
        sys.exit(1)
    with open('%s/main.toml' % cfg_dir, 'rb') as f:
        mod.__dict__.update(toml.load(f))
    for toml_path in glob('%s/*.toml' % cfg_dir):
        if toml_path.endswith('/config.toml'):
            continue
        with open(toml_path, 'rb') as f:
            k = os.path.basename(toml_path).replace('.toml', '')
            mod.__dict__[k] = toml.load(f)
    __expand_vars(mod.__dict__)


__load()
