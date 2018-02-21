#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from multiprocessing import Lock
import os
from subprocess import call, DEVNULL
import time

from pybess.bess import BESS
import resq.config as config
from resq.nf import NFInstance
from resq.util import Singleton
from resq.util.log import log_debug, log_error
from resq.util import PciDevice


class Port(object, metaclass=Singleton):
    def __init__(self, name):
        self.name = name
        self.numa_node = None
        self.hwaddr = None
        self.pciaddr = None
        self.neighbor_chassis = None
        self.neighbor_hwaddr = None
        self.neighbor_name = None
        self.neighbor_pciaddr = None
        self.dpdk_driver = None
        self.netmap_driver = None
        self.slaves = None
        self.speed = None

        self.__dict__.update(config.ports[name])
        self.dev = PciDevice(self.pciaddr) if not self.is_virtual else None

        if self.numa_node is None and self.dev:
            self.numa_node = self.dev.numa_node

        if not self.is_virtual:
            assert (self.dev.numa_node == self.numa_node)

        if not self.is_bond:
            assert (not self.slaves)

        if self.is_bond:
            assert (self.slaves and len(self.slaves) >= 1)
            self.bond_id = int(self.name.replace('eth_bond', ''))
            self.slaves = [Port(s) for s in self.slaves]

            numa_nodes = set(map(lambda s: s.numa_node, self.slaves))
            neighbor_numa_nodes = set(
                map(lambda s: s.neighbor_numa_node, self.slaves))
            assert (len(numa_nodes) == len(neighbor_numa_nodes) == 1)
            self.numa_node = numa_nodes.pop()
            self.neighbor_numa_node = neighbor_numa_nodes.pop()

            self.hwaddr = min([s.hwaddr for s in self.slaves])
            self.hwaddr = self.hwaddr[:-2] + \
                '%2x' % (int(self.hwaddr[-2:], 16) - 1)
            self.neighbor_hwaddr = \
                min([s.neighbor_hwaddr for s in self.slaves])
            self.neighbor_hwaddr = self.neighbor_hwaddr[:-2] + \
                '%2x' % (int(self.neighbor_hwaddr[-2:], 16) - 1)

            self.pciaddr = '%s,mode=0,socket_id=%d,mac=%s' \
                % (self.name, self.numa_node, self.hwaddr)
            self.neighbor_pciaddr = '%s,mode=0,socket_id=%d,mac=%s' \
                % (self.neighbor_name, self.neighbor_numa_node,
                   self.neighbor_hwaddr)
            for s in self.slaves:
                self.pciaddr += ',slave=0000:%s' % s.pciaddr
                self.neighbor_pciaddr += ',slave=0000:%s' % s.neighbor_pciaddr
        elif self.is_ring:
            assert (self.numa_node is not None)
            self.ring_id = int(self.name.replace('eth_ring', ''))
            self.pciaddr = '%s,nodeaction=ring%d:%d:CREATE' % (self.name,
                                                               self.ring_id,
                                                               self.numa_node)
            # TODO: neighbor
            # self.neighbor_pciaddr = '%s,nodeaction=ring%d:%d:CREATE' % (
            #         self.name, self.ring_id, self.numa_node)
        self.bess_registered = False

    def __str__(self):
        return 'Port(%s)' % (self.name)

    @property
    def driver(self):
        if self.is_bond:
            return 'pmd_bond'
        elif self.is_ring:
            return 'pmd_ring'
        elif self.dev:
            if hasattr(self.dev, 'driver'):
                return self.dev.driver
            else:
                return None
        return self.dev.driver if self.dev else None

    @property
    def driver_type(self):
        if self.is_bond or self.is_ring:
            return 'dpdk'
        elif self.driver == self.dpdk_driver:
            return 'dpdk'
        elif self.driver == self.netmap_driver:
            return 'netmap'

    @property
    def irq_list(self):
        if not self.dev:
            return []
        irqs = []
        with open('%s/irq' % self.dev.ddir, 'r') as f:
            irqs.append(int(f.read()))
        for fname in glob.glob('%s/msi_irqs/*' % self.dev.ddir):
            irqs.append(int(os.path.basename(fname)))
        return irqs

    @property
    def is_available(self):
        return PortManager().is_available(self)

    @property
    def is_bond(self):
        return 'eth_bond' in self.name

    @property
    def is_dpdk(self):
        return self.driver == self.dpdk_driver or \
            self.is_bond or self.is_ring

    @property
    def is_netmap(self):
        return self.driver == self.netmap_driver

    @property
    def is_ring(self):
        return 'eth_ring' in self.name

    @property
    def is_virtual(self):
        return self.is_bond or self.is_ring

    @property
    def neighbor_id(self):
        if self.is_bond or self.is_ring:
            return self.neighbor_name
        return self.neighbor_hwaddr

    def _configure_dpdk(self, cores):
        # TODO: error handling
        if not self.bess_registered:
            pm = PortManager()
            if self.is_bond:
                for p in self.slaves:
                    if not pm.bess_create_port(p):
                        return False
                    p.bess_registered = True
            if not pm.bess_create_port(self):
                return False
            self.bess_registered = True
            time.sleep(1)

        return True

    def _configure_netmap(self, cores):
        # TODO: error handling
        cpumask = '%016x' % sum([pow(2, c.name) for c in cores])
        cpumask = '%s,%s' % (cpumask[0:8], cpumask[8:16])
        cmds = [['ifconfig', self.name, 'down'],
                ['ethtool', '-A', self.name, 'tx', 'off', 'rx', 'off'], [
                    'ethtool', '-K', self.name, 'tx', 'off', 'rx', 'off', 'sg',
                    'off', 'tso', 'off', 'gso', 'off', 'gro', 'off'
                ], [
                    'ethtool', '-C', self.name, 'adaptive-tx', 'off',
                    'adaptive-rx', 'off', 'rx-usecs', '5'
                ], ['ethtool', '-G', self.name, 'rx', '256', 'tx',
                    '256'], ['ethtool', '-L', self.name, 'combined',
                             '1'], ['ifconfig', self.name, 'promisc', 'up']]

        for cmd in cmds:
            call(cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)

        for irq in self.irq_list:
            path = '/proc/irq/%d/smp_affinity' % irq
            if not os.path.exists(path):
                continue
            with open(path, 'w') as f:
                f.write(cpumask)

        for fname in glob.glob('/sys/class/net/%s/queues/*/*cpus' % self.name):
            with open(fname, 'w') as f:
                f.write(cpumask)

        return True

    def configure(self, port_type, cores):
        if self.is_bond:
            for p in self.slaves:
                if port_type != p.driver_type:
                    if not p.update_driver(port_type):
                        return False
        else:
            if port_type != self.driver_type:
                if not self.update_driver(port_type):
                    return False

        if self.is_dpdk:
            return self._configure_dpdk(cores)
        elif self.is_netmap:
            return self._configure_netmap(cores)

    def free(self):
        return PortManager().free(self)

    def reserve(self):
        return PortManager().reserve(self)

    def update_driver(self, driver_type):
        # no-op if the driver is not changing
        if not self.dev or self.driver_type == driver_type:
            return True

        if self.bess_registered:
            assert (self.driver_type == 'dpdk')
            pm = PortManager()
            if self.is_bond:
                for p in self.slaves:
                    pm.bess_destroy_port(p)
                    p.bess_registered = False
            pm.bess_destroy_port(self)
            self.bess_registered = False
            time.sleep(1)

        # bring the port down if it's in netmap mode
        if self.driver_type == 'netmap':
            if call(
                ['ifconfig', self.name, 'down'],
                    stdin=DEVNULL,
                    stdout=DEVNULL,
                    stderr=DEVNULL):
                log_debug('Failed to bring port %s down!' % self.name)
                return False

        driver = getattr(self, '%s_driver' % driver_type)
        if not self.dev.change_driver(driver):
            return False

        time.sleep(1)
        return True


class PortManager(NFInstance, metaclass=Singleton):
    def __init__(self):
        super().__init__(name='bess')
        self.bess = BESS()
        self._lock = Lock()
        self._store = {}
        self._reserved = set()
        for name in config.ports.keys():
            self.register(Port(name))

    def _connect(self):
        if not self.bess.is_connected():
            try:
                self.bess.connect()
            except Exception as e:
                log_error('Failed to connect to BESS: %s' % e)
                return False
        return True

    def bess_create_port(self, port):
        if not self._connect():
            return False
        arg = {
            'vdev' if port.is_virtual else 'pci': port.pciaddr,
            'num_inc_q': 1,
            'num_out_q': 1,
            'size_inc_q': config.io['nr_bufs_rx'],
            'size_out_q': config.io['nr_bufs_tx']
        }
        try:
            with self._lock:
                self.bess.create_port(
                    driver='PMDPort', name=port.name, arg=arg)
        except Exception as e:
            log_error('Failed to create BESS port %s: %s' % (port.name, e))
            return False

        return True

    def bess_destroy_port(self, port):
        if not self._connect():
            return False
        try:
            with self._lock:
                self.bess.destroy_port(port.name)
        except Exception as e:
            log_error('Failed to destroy BESS port %s: %s' % (port.name, e))
            return False

        return True

    def free(self, port):
        if port not in self._reserved:
            raise ValueError('Port %s is already free' % port.name)
        if port.is_bond:
            for s in port.slaves:
                s.free()
        self._reserved.remove(port)

    def is_available(self, port):
        if port in self._reserved:
            return False
        if port.is_bond:
            return all([p.is_available for p in port.slaves])
        return True

    def list(self,
             available=None,
             numa_node=None,
             ring=None,
             speed=None,
             retname=False):
        ports = sorted(
            self._store.items(), key=lambda i: i[1].order)
        #ports = sorted(
        #    self._store.items(), key=lambda i: (i[0][-1], i[0][:-1]))
        return [
            k if retname else v for k, v in ports
            if (available is None or v.is_available == available) and (
                numa_node is None or v.numa_node == numa_node) and (
                    ring is None or v.is_ring == ring) and (
                        speed is None or v.speed == speed)
        ]

    def list_names(self, *args, **kwargs):
        return self.list(*args, **kwargs, retname=True)

    def reserve(self, port):
        if port in self._reserved:
            raise ValueError('Port %s is already reserved' % port.name)
        if not port.is_available:
            raise ValueError('Port %s could not be reserved' % port.name)
        if port.is_bond:
            for p in port.slaves:
                p.reserve()
        self._reserved.add(port)

    def register(self, port):
        self._store[port.name] = port

    def udev_net_rules(self):
        rules = []
        for name, p in self.list():
            if p.is_ring or p.is_bond:
                continue
            rules.append('SUBSYSTEM=="net", ACTION=="add", DRIVERS=="?*", '
                         'ATTR{address}=="%s", ATTR{dev_id}=="0x0", '
                         'ATTR{type}=="1", KERNEL=="eth*", NAME="%s"' %
                         (p.pciaddr, name))
        return '\n'.join(rules)
