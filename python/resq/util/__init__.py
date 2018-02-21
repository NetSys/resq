#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import humanfriendly
import glob
import kmod
import os
import platform
from pyparsing import Forward, Word, Optional, Literal, nums
from subprocess import call, check_output, Popen, DEVNULL

import resq.config as config
from resq.util.log import log_error
from resq.util.msr import writemsr


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        name = None
        if len(kwargs) == 0 and len(args) == 1:
            name = args[0]
        elif 'name' in kwargs:
            name = kwargs['name']
        if cls not in cls._instances:
            cls._instances[cls] = {}
        if name not in cls._instances[cls]:
            cls._instances[cls][name] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls][name]


def __str_to_list(s):
    def range_to_list(s, loc, tokens):
        l = []
        if len(tokens) == 1:
            l.append(int(tokens[0]))
        else:
            l += list(range(int(tokens[0]), int(tokens[1]) + 1))
        return l

    expr = Forward()
    term = (Word(nums) + Optional(Literal('-').suppress() + Word(nums))
            ).setParseAction(range_to_list)
    expr << term + Optional(Literal(',').suppress() + expr)
    return expr.parseString(s, parseAll=True)


def cbm_to_ways(cbm):
    if isinstance(cbm, type('1')):
        cbm = int(cbm, 16)
    return bin(cbm).count('1')


def cbm_to_size(cbm):
    return ways_to_size(cbm_to_ways(cbm))


def ways_to_cbm(nr_ways, nr_ways_used=0, hex=False):
    r = (2**nr_ways - 1) << nr_ways_used
    if hex:
        r = hex(r)[2:]
    return r


def ways_to_size(nr_ways):
    return l3_way_size() * nr_ways


def disable_c_state():
    for fname in glob.glob(
            '/sys/devices/system/cpu/*/cpuidle/state[1-9]/disable'):
        with open(fname, 'w') as f:
            f.write('1')


def disable_core_frequency_scaling():
    with open('/sys/module/processor/parameters/ignore_ppc', 'w') as f:
        f.write('1')

    if not os.path.exists('/sys/devices/system/cpu/cpu0/cpufreq'):
        return

    max_freq = None
    with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq') as f:
        max_freq = f.read()

    # run at max frequency
    for fname in glob.glob(
            '/sys/devices/system/cpu/*/cpufreq/scaling_governor'):
        with open(fname, 'w') as f:
            f.write('performance')
    for fname in glob.glob(
            '/sys/devices/system/cpu/*/cpufreq/scaling_min_freq'):
        with open(fname, 'w') as f:
            f.write(max_freq)
    for fname in glob.glob(
            '/sys/devices/system/cpu/*/cpufreq/scaling_max_freq'):
        with open(fname, 'w') as f:
            f.write(max_freq)


def disable_nmi_watchdog():
    # doesn't belong here but anyways
    with open('/proc/sys/kernel/nmi_watchdog', 'w') as f:
        f.write('0')


def disable_p_state():
    if not os.path.exists('/sys/devices/system/cpu/intel_pstate'):
        return

    with open('/sys/devices/system/cpu/intel_pstate/no_turbo', 'w') as f:
        f.write('1')
    with open('/sys/devices/system/cpu/intel_pstate/max_perf_pct', 'w') as f:
        f.write('100')
    with open('/sys/devices/system/cpu/intel_pstate/min_perf_pct', 'w') as f:
        f.write('100')


def disable_uncore_frequency_scaling():
    km = kmod.Kmod()
    km.modprobe('msr')
    # source: "Energy Efficient Servers, Blueprints for Data Center Optimization" on Google play
    writemsr(0x620, 0x1d1d)
    #writemsr(0x620, 0x3f3f)


def configure_ddio():
    default_ways = [17, 18]
    new_ways = [19, 20]
    mask = ways_to_cbm(config.ddio['nr_ways'], config.ddio['skip_ways'])
    # source: "https://software.intel.com/en-us/forums/software-tuning-performance-optimization-platform-monitoring/topic/600913"
    writemsr(0xc8b, mask)


def configure_irq_affinity():
    # move irqs to the non-NF cores
    affinity = sum([pow(2, c) for n in list_nodes() for c in list_cores(n)
                    if n != config.run_socket_id])
    affinity = '%016x' % affinity
    affinity = '%s,%s' % (affinity[0:8], affinity[8:16])
    call(['service', 'irqbalance', 'stop'])
    with open('/proc/irq/default_smp_affinity', 'w') as f:
        f.write(affinity)
    for fname in glob.glob('/proc/irq/*/smp_affinity'):
        try:
            with open(fname, 'w') as f:
                f.write(affinity)
        except Exception as e:
            pass
    for fname in glob.glob('/sys/class/net/*/queues/*/*cpus'):
        try:
            with open(fname, 'w') as f:
                f.write(affinity)
        except Exception as e:
            pass


def configure_pcie():
    try:
        with open('/sys/module/pcie_aspm/parameters/policy', 'w') as f:
            f.write('performance')
    except Exception as e:
        pass
    # 8086:154a: X520-4
    # 8086:1558: X520-Q1
    # 8086:1583: XL710-QDA2
    for pci_id in ['8086:1583']:
        l = ['CAP_EXP+30.w=0x0003:0x000f', # Link Control 2 Register: Target Link Speed=8GT/s
             #'CAP_EXP+8.w=0x0040:0x00e0', # Device Control Register: Max Payload(7:5)=256 (010b)
             'CAP_EXP+8.w=0x5000:0x7000', # Device Control Register: Max Read Request Size(14:12)=4096 (101b)
             'CAP_EXP+10.w=0x0020:0x0023', # Link Control Register: ASPM=disabled, Retrain Link
            ]
        for i in l:
            call(['setpci', '-d', pci_id, i],
                 stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)


### DISCOVERY
def list_cores(node=None):
    if node != None:
        with open('/sys/devices/system/node/node%d/cpulist' % node) as f:
            return __str_to_list(f.read())
    else:
        with open('/sys/devices/system/cpu/online') as f:
            return __str_to_list(f.read())


def list_nodes():
    with open('/sys/devices/system/node/online') as f:
        return __str_to_list(f.read())


def nr_cores(node=None):
    return len(list_cores(node))


def nr_nodes():
    return len(list_nodes())


def l3_nr_ways():
    with open(
            '/sys/devices/system/cpu/cpu0/cache/index3/ways_of_associativity') as f:
        return int(f.read())


# in MB
def l3_size():
    with open('/sys/devices/system/cpu/cpu0/cache/index3/size') as f:
        return humanfriendly.parse_size(f.read()) / (1024. * 1024.)


# in MB
def l3_way_size():
    return l3_size() / l3_nr_ways()


def l3_flush():
    cmds = [['modprobe', '-r', 'wbinvd'], ['modprobe', 'wbinvd']]
    for cmd in cmds:
        if call(cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL):
            log_error('Failed to flush LLC: %s', ' '.join(cmd))


def l3_flush_stream(node):
    core_id = list_cores(node)[-1]
    proc = Popen(['cgexec', '-g', 'cpuset:/',
                  'taskset', '-c', str(core_id),
                  '/opt/tools/stream/stream'],
                 stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)
    resctrl_dir = '/sys/fs/resctrl/stream'
    if not os.path.exists(resctrl_dir):
        os.mkdir(resctrl_dir)
    l3cbm = 'L3:' + ';'.join(['%d=%s' %
                              (i, '%x' % config.cat['cbm_max']
                               if i == config.run_socket_id
                               else '%x' % config.cat['cbm_max'])
                              for i in list_nodes()])
    with open('%s/tasks' % resctrl_dir, 'w') as f:
        f.write(str(proc.pid))
    with open('%s/schemas' % resctrl_dir, 'w') as f:
        f.write(l3cbm)
    proc.wait()


def init_dpdk():
    km = kmod.Kmod()
    km.modprobe('uio')
    km.modprobe('uio_pci_generic')
    km.modprobe('igb_uio')
    km.modprobe('vfio-pci')


def init_hugepages():
    if not os.path.exists(config.huge_path):
        os.mkdir(config.huge_path)
    if not os.path.ismount(config.huge_path):
        call(['mount', '-t', 'hugetlbfs', '-o',
              'rw,relatime,pagesize=%dkB' % config.hugepage_size, 'nodev', config.huge_path],
             stdin=DEVNULL,
             stdout=DEVNULL,
             stderr=DEVNULL)
    with open('/proc/sys/kernel/shmmax', 'w') as f:
        f.write('1073741824')
    with open('/proc/sys/vm/hugetlb_shm_group', 'w') as f:
        f.write('0')
    for node in range(config.nr_nodes):
        with open('/sys/devices/system/node/node%d/hugepages/hugepages-%dkB/nr_hugepages'
                  % (node, config.hugepage_size), 'w') as f:
            f.write('%d' % (config.nr_hugepages / config.nr_nodes))

    # disable transparent huge pages to avoid surprise performance boosts
    with open('/sys/kernel/mm/transparent_hugepage/enabled', 'w') as f:
        f.write('never')
    with open('/sys/kernel/mm/transparent_hugepage/defrag', 'w') as f:
        f.write('never')


def init_lxc():
    cpus_lxc = [ str(x) for x in list_cores(config.run_socket_id) ]
    cpus_rest = [ str(x) for n in list_nodes() for x in list_cores(n) if n != config.run_socket_id ]
    with open('/sys/fs/cgroup/cpuset/cgroup.clone_children', 'w') as f:
        f.write('1')
    call(['cgcreate', '-g', 'cpuset:lxc'])
    call(['cgset', '-r', 'cpuset.cpus=%s' % ','.join(cpus_lxc), 'lxc'])
    call(['cgset', '-r', 'cpuset.cpu_exclusive=1', 'lxc'])
    call(['cgset', '-r', 'cpuset.mems=%s' % ','.join(str(x) for x in list_nodes()), 'lxc'])
    call(['cgcreate', '-g', 'cpuset:rest'])
    call(['cgset', '-r', 'cpuset.cpus=%s' % ','.join(cpus_rest) , 'rest'])
    call(['cgset', '-r', 'cpuset.cpu_exclusive=0', 'rest'])
    call(['cgset', '-r', 'cpuset.mems=%s' % ','.join(str(x) for x in list_nodes() if int(x) != config.run_socket_id) , 'rest'])
    tasks = [str(int(i)) for i in open('/sys/fs/cgroup/cpuset/tasks')]
    call([ 'cgclassify', '-g', 'cpuset:rest', *tasks ], stdout=DEVNULL, stderr=DEVNULL)
    call([ 'cgclassify', '-g', 'cpuset:/', '1', str(os.getpid()) ])

def init_netmap():
    core_id = str(list_cores(config.run_socket_id)[0])
    if call(['cgexec', '--sticky', '-g', 'cpuset:/', 'taskset', '-c', core_id,
             'modprobe', 'netmap'],
            stdin=DEVNULL,
            stdout=DEVNULL,
            stderr=DEVNULL):
        log_error('Failed to modprobe netmap')
        return

    for mod in ['i40e', 'ixgbe']:
        if os.path.exists('/sys/module/netmap/holders/%s' % mod):
            continue
        call(['modprobe', '-r', mod],
             stdin=DEVNULL,
             stdout=DEVNULL,
             stderr=DEVNULL)
        if call(['cgexec', '--sticky', '-g', 'cpuset:/', 'taskset', '-c',
                 core_id, 'modprobe', mod],
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL):
            log_error('Failed to modprobe %s' % mod)
            return
        if not os.path.exists('/sys/module/netmap/holders/%s' % mod):
            log_error('%s module is not netmap-enabled.')
            return

    with open('/sys/module/netmap/parameters/admode', 'w') as f:
        f.write('1')
    with open('/sys/module/netmap/parameters/buf_num', 'w') as f:
        f.write('16640')
        #f.write('12484')
    with open('/sys/module/netmap/parameters/buf_size', 'w') as f:
        f.write('2176')
    #with open('/sys/module/netmap/parameters/if_size', 'w') as f:
    #    f.write('2048')
    #with open('/sys/module/netmap/parameters/if_num', 'w') as f:
    #    f.write('17')
    #with open('/sys/module/netmap/parameters/ring_num', 'w') as f:
    #    f.write('128')
    #with open('/sys/module/netmap/parameters/ring_size', 'w') as f:
    #    f.write('36864')
    with open('/sys/module/i40e/parameters/ix_crcstrip', 'w') as f:
        f.write('1')


def init_rdt():
    resctrl_dir = '/sys/fs/resctrl'
    if not os.path.ismount(resctrl_dir):
        if call(['mount', '-t', 'resctrl', 'resctrl', '-o', 'verbose',
                 resctrl_dir],
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL):
            log_error('Failed to mount resctrl')
            return


def lspci_net():
    ETHERNET_PCI_CLASS = '0200'
    slots = []

    out = check_output('lspci', '-nd',
                       '::%s' % ETHERNET_PCI_CLASS).splitlines()
    for line in out:
        slots.append(line.split()[0])
    return slots


class PciDevice(object):
    def __init__(self, bus, dev=None, func=None):
        if not dev or not func:
            if bus.count(':') == 1:
                bus, dev = bus.split(':')
            elif bus.count(':') == 2:
                _, bus, dev = bus.split(':')
            else:
                raise ValueError('Invalid BDF value %s' % bus)
            bus = int(bus, 16)
            dev, func = [int(i, 16) for i in dev.split('.')]
        self.bus = bus
        self.dev = dev
        self.func = func
        self.bdf = '%02x:%02x.%01x' % (bus, dev, func)
        self.ddir = "/sys/bus/pci/devices/0000:%s" % self.bdf
        self.config_fn = "%s/config" % (self.ddir)
        self._readvars()

    def _readvars(self):
        out = check_output(['lspci', '-Dvmmnks', self.bdf]).splitlines()
        for line in out:
            if len(line) == 0:
                continue
            name, value = line.decode().split('\t', 1)
            setattr(self, name.rstrip(':').lower(), value)

    def change_driver(self, driver):
        self._readvars()
        with open('%s/driver_override' % self.ddir, 'w') as f:
            f.write(driver)

        if hasattr(self, 'driver'):
            if self.driver == driver:
                return False
            with open('%s/driver/unbind' % self.ddir, 'w') as f:
                f.write('0000:%s' % self.bdf)

        with open('/sys/bus/pci/drivers_probe', 'w') as f:
            f.write('0000:%s' % self.bdf)

        self._readvars()
        if self.driver != driver:
            log_error('Failed to bind %s to %s' % (self.bdf, driver))
            return False

        return True

    def disable(self):
        with open('%s/enable' % self.ddir, 'w') as f:
            f.write('0')
        self._readvars()

    def enable(self):
        with open('%s/enable' % self.ddir, 'w') as f:
            f.write('1')
        self._readvars()

    @property
    def numa_node(self):
        with open('%s/numa_node' % self.ddir) as f:
            return int(f.read())
        return -1

    def probe(self):
        return os.path.isfile(self.config_fn)


assert (platform.system() == 'Linux' and platform.machine() == 'x86_64')
