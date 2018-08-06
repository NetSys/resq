#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import enum
import multiprocessing.pool
from multiprocessing import Lock
import numpy as np
import os
import re
import shutil
import string
from subprocess import call, check_output, Popen, DEVNULL, PIPE
import time

import resq.config as config
from resq.util import Singleton
from resq.util import list_cores, list_nodes, nr_cores
from resq.util.log import log_error


class NF(object, metaclass=Singleton):
    def __init__(self, name):
        self.name = name
        self.cmd = None
        self.cmd_getpids = None
        self.cmd_start = None
        self.cmd_status = None
        self.cmd_stop = None
        self.desc = None
        self.desc_short = None
        self.measure_count_min = 0
        self.nr_cores = 1
        self.nr_ports = 0
        self.pkt_size_max = 1514
        self.port_type = None
        self.randomize_payload = False
        self.synthetic = False
        self.use_container = True
        self.__dict__.update(config.nfs[name])
        assert (self.cmd or self.cmd_getpids)
        assert (self.cmd or self.cmd_start)
        assert (self.cmd or self.cmd_status)
        assert (self.cmd or self.cmd_stop)
        if not self.cmd_getpids:
            self.cmd_getpids = self.cmd.replace('$CMD', 'getpids')
        if not self.cmd_start:
            self.cmd_start = self.cmd.replace('$CMD', 'start')
        if not self.cmd_status:
            self.cmd_status = self.cmd.replace('$CMD', 'status')
        if not self.cmd_stop:
            self.cmd_stop = self.cmd.replace('$CMD', 'stop')

    def __str__(self):
        return 'NF(%s)' % (self.name)

    @property
    def idle_cycles(self):
        return int(self.name[5:]) if not self.is_real else -1

    @property
    def is_mlc(self):
        return 'mlc' in self.name

    @property
    def is_real(self):
        return not self.is_mlc and not self.is_syn

    @property
    def is_syn(self):
        return 'syn' in self.name


class NFManager(object, metaclass=Singleton):
    def __init__(self):
        self._store = {}
        for name in config.nfs.keys():
            self.register(NF(name))

    def list(self, mlc=False, syn=False, real=False, retname=False):
        return [
            name if retname else nf for name, nf in self._store.items()
            if (syn and nf.is_syn) or (mlc and nf.is_mlc) or (
                real and nf.is_real and nf.name != 'dpdk_primary')
        ]

    def list_names(self, *args, **kwargs):
        return self.list(*args, **kwargs, retname=True)

    def register(self, nf):
        self._store[nf.name] = nf


class NFStatus(enum.Enum):
    unknown = -1
    stopped = 0
    running = 1
    ready = 2
    frozen = 3    # not in use


class NFInstance(object):
    def __init__(self, name, cores=[], ports=[], cbm=None, args=''):
        self.__dict__.update(NF(name).__dict__)
        self.cbm = cbm
        self.cores = tuple(cores)
        self.ports = tuple(ports)
        # FIXME: workaround for lxc-cgroup not taking effect with
        #        large number of containers/cgroups
        self.handle = 'resq-%s' % (self.ports[0].name
                                   if len(self.ports) > 0 else self.name)
        self.proc = None
        self.measure_count = 0
        self.args = args

    def __eq__(self, d):
        assert (False)

    def __getattr__(self, attr):
        assert (attr not in self.__dict__)
        if attr == '__getstate__':
            raise AttributeError

        if 'pkt_stats' in self.__dict__ and attr in self.pkt_stats:
            return self.pkt_stats[attr]

        if attr[0:3] == 'rtt':
            return 5000

        if 'perf_stats' in self.__dict__:
            if attr != 'llc_occupancy':
                attr = attr.replace('_', '-')
            if attr in self.perf_stats:
                return self.perf_stats[attr]

        # a hack for incomplete runs
        if 'pps' in attr:
            return 0

        raise AttributeError

    def __ne__(self, e):
        return not self.__eq__(e)

    def _prep_cmd(self, s):
        # FIXME: %PORTS% broken for dpdk vdevs (which may have commas in name)
        env = {
            'HANDLE': self.handle,
            'PORTS': ','.join(str(x.name) for x in self.ports),
            'CORES': ','.join(str(x.name) for x in self.cores),
            'ARGS': self.args
        }
        for i, port in enumerate(self.ports):
            env['PORT%d' % i] = port.name
            env['PCIADDR%d' % i] = port.pciaddr
        for i, core in enumerate(self.cores):
            env['CORE%d' % i] = str(core.name)
        cmd = []
        if self.use_container:
            cmd += ['lxc-attach', '-n', self.handle, '--']
        cmd += string.Template(s).substitute(env).split()
        return cmd

    @property
    def is_done(self):
        return hasattr(self, 'rx_mbps_mean')

    @property
    def status(self):
        cmd = self._prep_cmd(self.cmd_status)

        try:
            status = check_output(
                cmd, stdin=DEVNULL, stderr=DEVNULL).decode().strip()
        except Exception as e:
            return NFStatus.stopped

        if status == 'RUNNING':
            return NFStatus.running
        elif status == 'STOPPED':
            return NFStatus.stopped
        elif status == 'READY':
            return NFStatus.ready

        return NFStatus.unknown

    def enforce_limits(self):
        # assert (self.use_container and self.cbm)
        cpus_list = ','.join(str(c.name) for c in self.cores)
        l3cbm = 'L3:' + ';'.join([
            '%d=%s' %
            (i, '%x' % self.cbm
             if i == config.run_socket_id else '%x' % config.cat['cbm_max'])
            for i in list_nodes()
        ])
        resctrl_dir = '/sys/fs/resctrl/%s' % \
                      l3cbm.replace(':', '__').replace(';', '_')
        with NFInstanceManager()._lock:
            if not os.path.exists(resctrl_dir):
                os.mkdir(resctrl_dir)
            else:
                with open('%s/cpus_list' % resctrl_dir) as f:
                    cpus_list += ',' + f.read()
            with open('%s/cpus_list' % resctrl_dir, 'w') as f:
                f.write(cpus_list)
            with open('%s/schemata' % resctrl_dir, 'w') as f:
                f.write(l3cbm + '\n')
            with open('%s/schemata' % resctrl_dir) as f:
                import re
                pattern = r'=0*'
                retval = f.read().strip()
                if re.sub(pattern, '=', l3cbm) != re.sub(pattern, '=', retval):
                    log_error(
                        '%s: Failed to update cache allocation: %s instead of %s.'
                        % (self.handle, retval, l3cbm))
                    return False
        if self.use_container and call(
            [
                'lxc-cgroup', '-n', self.handle, 'cpuset.cpus', ','.join(
                    str(x.name) for x in self.cores)
            ],
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL):
            log_error('%s: Failed to set cpu restrictions.' % self.handle)
            return False
        if self.use_container and call(
            ['lxc-cgroup', '-n', self.handle, 'cpuset.cpu_exclusive', '1'],
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL):
            log_error('%s: Failed to set cpu exclusivity.' % self.handle)
            return False
        return True

    def free(self):
        for core in self.cores:
            core.free()
        for port in self.ports:
            port.free()

    def getpids(self):
        cmd = self._prep_cmd(self.cmd_getpids)

        try:
            pids = check_output(
                cmd, stdin=DEVNULL, stderr=DEVNULL).decode().strip()
        except Exception as e:
            return None

        return [int(x) for x in pids.split()]

    def monitor_pmu(self,
                    warmup_sec=config.warmup_sec,
                    duration_sec=config.duration_sec):
        return asyncio.get_event_loop().run_until_complete(
            self.monitor_pmu_async(warmup_sec, duration_sec))

    @asyncio.coroutine
    def monitor_pmu_async(self,
                          warmup_sec=config.warmup_sec,
                          duration_sec=config.duration_sec):
        cmd = [
            'perf', 'stat', '--field-separator', ',', '--delay',
            str(warmup_sec), '--repeat',
            str(duration_sec - warmup_sec), '--event',
            ','.join(config.perf_events['core']), '--cpu', ','.join(
                str(x.name) for x in self.cores), '--', 'sleep', '1'
        ]
        stats = {}

        proc = yield from asyncio.create_subprocess_exec(
            *cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=PIPE)
        out = yield from proc.stderr.read()
        for line in out.decode().splitlines():
            value, unit, name, *etc = line.split(',')
            name = re.sub(r'^[^/]*/', '', name).\
                replace(':pp', '').\
                replace('/', '').\
                replace('-', '_')
            if unit == 'MB':
                value = str(1000 * 1000 * float(value))
            stats[name] = 0 if 'not counted' in value else int(float(value))

        yield from proc.wait()
        return stats

    def resume(self):
        if not self.use_container:
            raise Exception

        call(
            ['lxc-unfreeze', '-n', self.handle],
            stdin=DEVNULL,
            stdout=DEVNULL,
            stderr=DEVNULL)

    def start(self):
        if self.status == NFStatus.ready:
            if self.use_container:
                if call(
                    [
                        'lxc-cgroup', '-n', self.handle,
                        'cpuset.cpu_exclusive', '0'
                    ],
                        stdin=DEVNULL,
                        stdout=DEVNULL,
                        stderr=DEVNULL):
                    log_error(
                        '%s: Failed to unset cpu exclusivity.' % self.handle)
                    return
            return

        for port in self.ports:
            if not port.configure(self.port_type, self.cores):
                return False

        if self.use_container:
            # prepare lxc directory
            lxc_dir = '/var/lib/lxc/%s' % self.handle
            if not os.path.isdir(lxc_dir):
                os.mkdir(lxc_dir)
            config_file = '%s/config' % lxc_dir
            try:
                shutil.copy(config.lxc_cfg, config_file)
            except Exception as e:
                log_error('%s: Failed to copy LXC cfg: %s' % (self.handle, e))
            if not os.path.isdir(config.lxc_rootfs):
                os.mkdir(config.lxc_rootfs)
                for i in ['sys', 'root']:
                    os.mkdir('%s/%s' % (config.lxc_rootfs, i))

            cores_str = ','.join(str(x.name) for x in self.cores)
            cmd = [
                'lxc-start', '-n', self.handle, '-f', config.lxc_cfg,
                '-s', 'lxc.cgroup.cpuset.mems=%s' % config.run_socket_id,
                '-s', 'lxc.cgroup.cpuset.cpus=%s' % cores_str
            ]
            for port in self.ports:
                if os.path.isdir('/sys/class/net/%s' % port.name):
                    cmd += [
                        '-s', 'lxc.network.type=phys', '-s',
                        'lxc.network.flags=up', '-s',
                        'lxc.network.link=%s' % port.name
                    ]
            if call(cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL):
                log_error('%s: Failed to start container.' % self.handle)
                return
        cmd = ['cgexec', '--sticky', '-g', 'cpuset:/']
        cmd += self._prep_cmd(self.cmd_start)
        self.proc = Popen(cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)

    def stop(self):
        if self.use_container:
            call(
                ['lxc-stop', '-n', self.handle],
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL)
        else:
            call(
                self._prep_cmd(self.cmd_stop),
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL)
            if self.proc:
                self.proc.poll()
                if self.proc.returncode is None:
                    self.proc.terminate()
        if self.proc:
            self.proc.wait()
            self.proc = None

        resctrl_dir = '/sys/fs/resctrl/%s' % self.handle
        if os.path.exists(resctrl_dir):
            os.rmdir(resctrl_dir)

    def suspend(self):
        if not self.use_container:
            raise Exception
        if self.status == NFStatus.running:
            call(
                ['lxc-freeze', '-n', self.handle],
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL)

    def wait(self, timeout):
        t = time.time()
        while time.time() - t < timeout:
            if self.proc:
                self.proc.poll()
                if self.proc.returncode is not None:
                    self.proc.wait()
                    self.proc = None
            status = self.status
            if status == NFStatus.ready:
                return True
            if not self.proc and status != NFStatus.running:
                return False
            time.sleep(0.25)

        return False


class NFInstanceManager(object, metaclass=Singleton):
    def __init__(self):
        from resq.cpu import CoreManager
        from resq.port import PortManager
        self._lock = Lock()
        self._instances_active = []
        self._instances_standby = []
        self._pool = multiprocessing.pool.ThreadPool(
            nr_cores(config.run_socket_id))
        self.core_manager = CoreManager()
        self.port_manager = PortManager()

    def _reserve_cores(self, n):
        cores = self.core_manager.list(
            available=True, numa_node=config.run_socket_id)
        if len(cores) < n:
            return None
        for c in cores[:n]:
            c.reserve()
        return cores[:n]

    def _reserve_ports(self, n, speed):
        ports = self.port_manager.list(
            available=True, numa_node=config.run_socket_id, speed=speed)
        if len(ports) < n:
            return None
        for p in ports[:n]:
            p.reserve()
        return ports[:n]

    def enforce_limits(self):
        if not all(
                self._pool.map(lambda i: i.enforce_limits(),
                               self._instances_active)):
            return False
        # give the rest of the cache to inactive cores
        active_cores = set(
            [c.name for i in self._instances_active for c in i.cores])
        inactive_cores = [
            str(c) for c in list_cores(config.run_socket_id)
            if c not in active_cores
        ]
        cpus_list = ','.join(inactive_cores)
        resctrl_dir = '/sys/fs/resctrl/node%d' % config.run_socket_id
        active_cbm = np.bitwise_or.reduce(
            [i.cbm for i in self._instances_active])
        inactive_cbm = config.cat['cbm_max'] - active_cbm
        if inactive_cbm == 0:
            inactive_cbm = config.cat['cbm_max']
        l3cbm = 'L3:' + ';'.join([
            '%d=%s' %
            (i, '%x' % inactive_cbm
             if i == config.run_socket_id else '%x' % config.cat['cbm_max'])
            for i in list_nodes()
        ])
        if not os.path.exists(resctrl_dir):
            os.mkdir(resctrl_dir)
        with open('%s/cpus_list' % resctrl_dir, 'w') as f:
            f.write(cpus_list)
        with open('%s/schemata' % resctrl_dir, 'w') as f:
            f.write(l3cbm + '\n')
        with open('%s/schemata' % resctrl_dir) as f:
            import re
            pattern = r'=0*'
            retval = f.read().strip()
            if re.sub(pattern, '=', l3cbm) != re.sub(pattern, '=', retval):
                log_error(
                    'Failed to update cache allocation: %s instead of %s.' %
                    (retval, l3cbm))
                return False
        return True

    def create_instances(self, run):
        # TODO: suspend active instances and save for a future run
        self._instances_standby += [i for i in self._instances_active if i]
        self._instances_active = [None] * run.nr_pipelets

        # resource demand
        nr_ports = {'dpdk': 0, 'netmap': 0}
        for name, speed in zip(run.pipelets, run.port_speeds):
            nf = NF(name)
            nr_ports[nf.port_type] += NF(name).nr_ports

        # FIXME: for now kill of any instance using eth_bond0 ports if needed
        ensure_free_ports = []
        # if run.port_speeds[0] == 40000:
        #     from resq.port import Port
        #     ensure_free_ports += Port('eth_bond0').slaves

        # search among standby instances and reuse if possible
        for i, (name, speed, cbm) in \
                enumerate(zip(run.pipelets, run.port_speeds, run.cbms)):
            nf = NF(name)
            for j, instance in enumerate(self._instances_standby):
                if instance.name == name and \
                        all([p.speed == speed and p not in ensure_free_ports
                             for p in instance.ports]):
                    self._instances_active[i] = instance
                    instance.cbm = cbm
                    del self._instances_standby[j]
                    break

        # kill standby instances
        all(self._pool.map(lambda i: i.stop(), self._instances_standby))
        all(self._pool.map(lambda i: i.free(), self._instances_standby))

        self._instances_standby = []

        # FIXME: assumes input is admissible
        for i, (name, speed, cbm) in \
                enumerate(zip(run.pipelets, run.port_speeds, run.cbms)):
            if self._instances_active[i]:
                continue

            nf = NF(name)
            cores = self._reserve_cores(n=nf.nr_cores)
            if not cores:
                pass

            ports = []
            if nf.nr_ports > 0:
                ports = self._reserve_ports(n=nf.nr_ports, speed=speed)
                if not ports:
                    log_error('Could not reserve ports for %s (NF #%d)' %
                              (name, i))
                    for core in cores:
                        core.free()
                    return False

            self._instances_active[i] = NFInstance(
                name=name, cores=cores, ports=ports, cbm=cbm)

        from copy import copy
        run._instances = copy(self._instances_active)
        return True

    def list(self):
        return self._instances_active

    def start(self):
        all(self._pool.map(lambda i: i.start(), self._instances_active))

    def stop(self):
        all(self._pool.map(lambda i: i.stop(), self._instances_active))
        all(self._pool.map(lambda i: i.stop(), self._instances_standby))
        self._pool.close()
        self._pool.join()

    def wait(self, timeout):
        return all(
            self._pool.map(lambda i: i.wait(timeout), self._instances_active))


# FIXME: for now, I only support single NF pipelets
class Pipelet(object):
    def __init__(self, l):
        self.nfs = l
