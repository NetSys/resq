#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import asyncio
import atexit
from collections import Iterable, defaultdict
import functools
import hashlib
import itertools as it
import numpy as np
import operator
from pprint import pformat
import re
from sqlalchemy import Column, Integer, PickleType, orm
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy_utils import ScalarListType
from subprocess import call, DEVNULL, PIPE

import resq.config as config
from resq.nf import NF, NFStatus
from resq.traffic import Traffic
from resq.traffic.generator import MelvinGen
from resq.util import Singleton
import resq.util.db as db
from resq.util.log import log_debug, log_error, log_info
from resq.util import cbm_to_size, cbm_to_ways, ways_to_cbm, disable_c_state, disable_p_state, disable_nmi_watchdog, disable_core_frequency_scaling, disable_uncore_frequency_scaling, init_dpdk, init_hugepages, init_lxc, init_netmap, init_rdt, l3_flush, l3_flush_stream, configure_ddio, configure_irq_affinity, configure_pcie


class Run(db.DeclarativeBase):
    __tablename__ = 'run'
    instances = {}
    run_id = Column(Integer(), primary_key=True)
    pipelets = Column(ScalarListType(), nullable=False)
    cbms = Column(ScalarListType(int), nullable=False)
    traffics = Column(ScalarListType(), nullable=False, default=False)
    utilizations = Column(ScalarListType(int), nullable=False, default=False)
    run_number = Column(Integer, nullable=False)
    results = Column(PickleType, nullable=False)

    def __new__(cls,
                pipelets=None,
                cbms=None,
                traffics=None,
                utilizations=None,
                run_number=0):
        if pipelets is None:
            return super().__new__(cls)

        assert (pipelets is not None)
        for i in [pipelets, cbms, traffics, utilizations]:
            assert (i is None or isinstance(i, Iterable))
        #assert(len(set(cbms)) <= 4)
        #assert(len(set(cbms)) == len(list(it.groupby(cbms))))

        if db.session is None:
            db.init()

        nr_pipelets = len(pipelets)

        if cbms is None:
            cbms = (config.cat['cbm_max'], ) * nr_pipelets
        if traffics is None:
            traffics = ('u100k_60', ) * nr_pipelets
        if utilizations is None:
            utilizations = (100, ) * nr_pipelets
        run_number = run_number

        # TODO: remove workaround and fix the test logic
        traffics = list(traffics)
        for i, (nf, traffic) in enumerate(zip(pipelets, traffics)):
            pkt_size_max = NF(nf).pkt_size_max
            pkt_size = Traffic(traffic).size[0][0]
            if pkt_size > pkt_size_max:
                traffics[i] = traffic.replace(str(pkt_size), str(pkt_size_max))
        traffics = tuple(traffics)

        key = (tuple(pipelets), tuple(cbms), tuple(traffics),
               tuple(utilizations), run_number)
        if key in Run.instances:
            return Run.instances[key]

        obj = db.session.query(Run).filter_by(
            pipelets=pipelets,
            cbms=cbms,
            traffics=traffics,
            utilizations=utilizations,
            run_number=run_number).first()

        if obj is None:
            obj = super().__new__(cls)

        Run.instances[key] = obj

        return obj

    def __init__(self,
                 pipelets,
                 cbms=None,
                 traffics=None,
                 utilizations=None,
                 run_number=0):
        if self.results is not None:
            return

        nr_pipelets = len(pipelets)

        if cbms is None:
            cbms = (config.cat['cbm_max'], ) * nr_pipelets
        if traffics is None:
            traffics = ('u100k_60', ) * nr_pipelets
        if utilizations is None:
            utilizations = (100, ) * nr_pipelets

        self.pipelets = pipelets
        self.cbms = cbms
        self.traffics = traffics
        self.utilizations = utilizations
        self.run_number = run_number
        self.results = {}
        self.__normalizer__()

    @orm.reconstructor
    def __normalizer__(self):
        self.pipelets = np.array(self.pipelets)
        self.cbms = np.array(self.cbms)
        self.traffics = np.array(self.traffics)
        self.utilizations = np.array(self.utilizations)

    def __eq__(self, e):
        return (isinstance(e, self.__class__) and all([
            hasattr(e, k) and getattr(e, k) == v for k, v in self.__dict__
            if k != 'results'
        ]))

    def __getattr__(self, attr):
        if attr in ['__getstate__', '__setstate__']:
            raise AttributeError
        if 'results' in self.__dict__ and attr in self.results:
            if isinstance(self.results[attr], Iterable):
                return np.array(self.results[attr])
            return self.results[attr]
        if '_solos_' in attr:
            subattr = attr[:attr.index('_solos')]
            solos = getattr(self, attr[attr.index('solos_'):])
            #return np.vectorize(lambda i: getattr(i, subattr)[0])(solos)
            return np.array([
                getattr(s, subattr)[0] if hasattr(s, subattr) else np.nan
                for s in solos
            ])
        if '_normalized_' in attr:
            subattr = attr[:attr.index('_normalized')]
            self_val = getattr(self, subattr)
            solo_val = getattr(self, attr.replace('normalized', 'solos'))
            return 100 * self_val / solo_val
        if attr in ['idle_cycles', 'is_mlc', 'is_real', 'is_syn']:
            return np.array([getattr(NF(i), attr) for i in self.pipelets])

        raise AttributeError(attr)

    def __hash__(self):
        return int(hashlib.sha1(str(self).encode()).hexdigest(), 16)

    def __ne__(self, e):
        return not self.__eq__(e)

    def __str__(self):
        return 'Run(%s)' % ','.join([
            '%s=%s' % (k, getattr(self, k))
            for k in
            ['pipelets', 'cbms', 'traffics', 'utilizations', 'run_number']
        ])

    def copy(self, **kwargs):
        for name, value in self.__dict__.items():
            if name in ['pipelets', 'cbms', 'traffics', 'utilizations', 'run_number'] and \
                    name not in kwargs:
                kwargs[name] = value
        return Run(**kwargs)

    def encode(self, encoding):
        return str(self).encode(encoding)

    @property
    def all_cbms(self):
        f = filter(
            lambda e: e.is_done, [
                self.copy(
                    cbms=tuple([ways_to_cbm(nr_ways)] + list(self.cbms[1:])))
                for nr_ways in range(config.cat['nr_ways_min'],
                                     config.cat['nr_ways_max'] + 1)
            ])
        return np.array(list(f))

    @property
    def all_runs(self):
        f = filter(
            lambda e: e.is_done,
            [self.copy(run_number=r) for r in config.run_numbers])
        return np.array(list(f))

    @property
    def all_runs_pending(self):
        f = filter(
            lambda e: not e.is_done,
            [self.copy(run_number=r) for r in config.run_numbers])
        return np.array(list(f))

    @property
    def all_utils(self):
        f = filter(
            lambda e: e.is_done, [
                self.copy(
                    utilizations=tuple([u] + list(self.utilizations[1:])))
                for u in config.utilizations
            ])
        return np.array(list(f))

    @property
    def has_solos(self):
        return all(self.solos_full_utilization)

    @property
    def is_done(self):
        try:
            return self.is_sane()[0]
        except Exception as e:
            log_error('failed to test for run sanity: %s' % e)
            return False

    @property
    def is_necessary(self):
        u = self.utilizations[0]
        if u in [min(config.utilizations), max(config.utilizations)]:
            return True
        if u < min(config.utilizations) or u > max(config.utilizations):
            return False

        l = [e for e in self.all_utils if e.utilizations[0] != u]
        if len(l) < 2:
            return True
        l.sort(key=lambda e: e.utilizations[0])
        data = np.array([(e.utilizations[0], e.rtt_95[0], e) for e in l])
        idx = np.searchsorted(data.T[0], u)
        if idx in [0, len(data)]:
            return True
        rtt1, rtt2 = data[idx - 1][1], data[idx][1]
        if rtt2 + 5 < rtt1:
            return True
        return (rtt2 - rtt1) > 10

    @property
    def is_isolated(self):
        if self.is_mlc.any():
            return False
        int_cmasks = [int(i, 16) for i in self.cbms]
        for i, cmask in enumerate(int_cmasks):
            others = int_cmasks[:]
            others.pop(i)
            if others:
                or_ = functools.reduce(operator.or_, others)
                if cmask & or_ != 0:
                    return False
        return True

    @property
    def has_results(self):
        return 'rx_mbps_mean' in self.results and \
            'cache_references' in self.results

    def is_sane(self):
        assert (isinstance(self.pipelets, np.ndarray))

        if not self.has_results:
            return (False, ('no results', ))

        valid_rx = \
            np.any([self.is_mlc,
                    self.utilizations < 10,
                    self.mpps > 0.005], axis=0).all()
        if not valid_rx:
            RuntimeManager().tgen.stop()
        if self.nr_pipelets > 1 and 'syn' in self.pipelets[1]:
            return (valid_rx, ('invalid rx'))
        valid_rx_cv = \
            valid_rx and \
            np.any([self.is_mlc,
                    (self.rx_mpps_std / self.mpps) < 0.1], axis=0).all()
        valid_tx = \
            np.any([self.is_mlc,
                    self.utilizations == 100,
                    np.isclose(self.tx_mbps_request,
                               self.tx_mbps_mean,
                               atol=1, rtol=1e-2)], axis=0).all()
        valid_rxtx = \
            valid_rx and valid_tx and \
            np.any([self.utilizations > 80,
                    self.is_mlc,
                    np.isclose(self.mpps,
                               self.tx_mpps_mean,
                               atol=1e-2, rtol=2e-2)], axis=0).all()
        valid_rtt = \
            np.any([self.utilizations > 80,
                    self.is_mlc,
                    self.rtt_95 < 100], axis=0).all() and \
            np.any([self.utilizations >= 95,
                    self.is_mlc,
                    self.rtt_95 < 300], axis=0).all()
        valid_utilization = \
            np.any([self.utilizations == 100,
                    self.is_mlc,
                    np.isclose(self.utilizations,
                               self.pps_normalized_full_utilization,
                               atol=1, rtol=3e-2)], axis=0).all()

        tests = [
            (valid_rx, 'invalid rx'),
            (valid_rx_cv, 'invalid rx cv'),
            (valid_tx, 'invalid tx'),
            (valid_rxtx, 'invalid rx/tx'),
            (valid_rtt, 'invalid rtt'),
            (valid_utilization, 'invalid utilization %s' %
             self.pps_normalized_full_utilization[0]),
        ]
        zipped = tuple(zip(*tests))
        retval = (all(zipped[0]),
                  tuple(it.compress(zipped[1], np.logical_not(zipped[0]))))
        return retval

    # cycles per packet
    @property
    def cpp(self):
        return self.cpu_cycles / self.pps

    # instructions per cycle
    @property
    def ipc(self):
        return self.instructions / self.cpu_cycles

    # instructions per packet
    @property
    def ipp(self):
        return self.instructions / self.pps

    @property
    def l3missrate(self):
        return 100 * self.cache_misses / self.cache_references

    # l3refs per packet
    @property
    def l3pp(self):
        return self.cache_references / self.pps

    @property
    def mpps(self):
        return self.rx_mpps_mean

    @property
    def pps(self):
        return 1e6 * self.rx_mpps_mean

    @property
    def loss_rate(self):
        return 100 * (
            self.tx_mpps_mean - self.rx_mpps_mean) / self.tx_mpps_mean

    @property
    def l3_sizes(self):
        return np.array([cbm_to_size(cbm) for cbm in self.cbms])

    @property
    def l3_ways(self):
        return np.array([cbm_to_ways(cbm) for cbm in self.cbms])

    @property
    def best_run(self):
        runs = self.all_runs
        if len(runs) == 0:
            return None
        elif len(runs) == 1:
            return runs[0]
        pps = [i.pps[0] for i in runs]
        return runs[np.argmax(pps)]

    @property
    def median_run(self):
        runs = self.all_runs
        if len(runs) == 0:
            return self
        elif len(runs) == 1:
            return runs[0]
        pps = np.vectorize(lambda e: e.pps[0])(runs)
        idx_pps = np.argmin(np.abs(pps - np.median(pps)))
        #rtt = np.vectorize(lambda e: e.rtt_95[0])(runs)
        #idx_rtt = np.argmin(np.abs(rtt - np.median(rtt)))
        return runs[idx_pps]

    @property
    def nr_cache_classes(self):
        return len(list(it.groupby(self.cbms)))

    @property
    def nr_pipelets(self):
        return len(self.pipelets)

    @property
    def pps_predict_cat(self):
        #assert(self.is_isolated)
        return self.pps_solos_same_cbm

    @property
    def pps_predict_nsdi12(self):
        from resq.profile import Profile
        if any(self.utilizations != 100):
            return [-1] * self.nr_pipelets
        pps_list = []
        try:
            if self.nr_pipelets == 1:
                return self.pps

            # an estimate for the number of llc refs
            syns_l = []
            for nf, cbm, traffic in \
                    zip(self.pipelets,
                        self.cbms,
                        self.traffics):
                syns_l.append(
                    Profile(nf, traffic).runs_l3(
                        other_nf_types=['syn'], isolated=False, is_done=True) +
                    [Run(pipelets=(nf, ), cbms=(cbm, ), utilizations=(100, ))])

            refs = self.cache_references_solos_same_cbm

            for index, app in enumerate(self.pipelets):
                other_refs = sum(refs) - refs[index]
                syns = syns_l[index]
                syns.sort(key=lambda e: sum(e.cache_references[1:]))
                idx1 = np.argmin([
                    np.abs(sum(e.cache_references[1:]) - other_refs)
                    for e in syns
                ])
                idx2 = idx1 + 1 if sum(
                    syns[idx1].cache_references[1:]) < other_refs else idx1 - 1
                if idx2 >= len(syns) or idx2 < 0:
                    pps_list.append(-1)
                    continue
                pps1, pps2 = syns[idx1].pps[0], syns[idx2].pps[0]
                ref1, ref2 = sum(syns[idx1].cache_references[1:]), sum(
                    syns[idx2].cache_references[1:])
                coeff = np.polyfit((ref1, ref2), (pps1, pps2), deg=1)
                ppps = np.polyval(coeff, other_refs)
                pps_list.append(ppps)
        except Exception as e:
            import sys
            log_error(e)
            log_error('line {}'.format(sys.exc_info()[-1].tb_lineno))

        return np.array(pps_list)

    @property
    def port_speeds(self):
        # FIXME: no 40g with netmap yet
        return np.array([
            40000
    #if i == 0 and NF(self.pipelets[i]).port_type == 'dpdk'
    #and Traffic(self.traffics[i]).size[0][0] > 60
    #else 10000
            for i in range(self.nr_pipelets)
        ])

    @property
    def solos_full_utilization(self):
        l = []
        for pipelet, nr_ways, traffic_spec in \
                zip(self.pipelets, self.l3_ways, self.traffics):
            e = Run(
                pipelets=(pipelet, ),
                cbms=(ways_to_cbm(nr_ways), ),
                run_number=self.run_number,
                traffics=(traffic_spec, ),
                utilizations=(100, ))
            l.append(e)
        return np.array(l)

    @property
    def solos_full_cbm(self):
        l = []
        for pipelet, cbm, traffic_spec, utilization in \
                zip(self.pipelets, self.cbms, self.traffics, self.utilizations):
            e = Run(
                pipelets=(pipelet, ),
                cbms=(config.cat['cbm_max'], ),
                run_number=self.run_number,
                traffics=(traffic_spec, ),
                utilizations=(utilization, ))
            l.append(e)
        return np.array(l)

    @property
    def solos_same_cbm(self):
        l = []
        for pipelet, nr_ways, traffic_spec, utilization in \
                zip(self.pipelets, self.l3_ways, self.traffics, self.utilizations):
            e = Run(
                pipelets=(pipelet, ),
                cbms=(ways_to_cbm(nr_ways), ),
                run_number=self.run_number,
                traffics=(traffic_spec, ),
                utilizations=(utilization, ))
            l.append(e)
        return np.array(l)

    @property
    def tx_mbps_request(self):
        l = []
        e = self.copy(utilizations=tuple([100] * self.nr_pipelets))
        for i, (u, t, nf, speed) in \
                enumerate(zip(self.utilizations, self.traffics, self.pipelets,
                              self.port_speeds)):
            pkt_size = Traffic(t).size[0][0]
            # FIXME: find a better way
            l.append(speed if u == 100 else -1 if not e.is_done else
                     e.rx_mpps_mean[i] * pkt_size * 8 * u / 100)
            #e.rx_mbps_mean[i] * u / 100)
        return np.array(l)

    # minimum cache allocation that yields a given normalized througphput
    def min_l3ways(self, pps_normalized):
        assert (self.nr_pipelets == 1)
        runs = self.all_cbms
        xput = [
            e.pps_normalized_full_cbm[0] - pps_normalized
            if e.pps_normalized_full_cbm[0] > pps_normalized else 100
            for e in runs
        ]
        idx = np.argmin(xput)
        return cbm_to_ways(runs[idx].cbms[0])

    def next_utilization(self, rtt=80):
        #self.cbms[0] == '3' and \
        if self.utilizations[0] == 100 and \
                self.is_real.all():
            next_util = self.rtt_to_utilization(rtt)
            if next_util:
                utilizations = tuple([next_util] + [100] *
                                     (len(self.utilizations) - 1))
                return self.copy(utilizations=utilizations)
        return None

    @functools.lru_cache()
    def optimal_calloc(self, nr_ways, policy):
        def neighbors():
            # per class min and max for number of ways
            nr_ways_min = config.cat['nr_ways_min']
            nr_ways_max = nr_ways - nr_ways_min * len(self.pipelets)

            e = self.copy(
                pipelets=self.pipelets[1:],
                cbms=self.cbms[1:],
                traffics=self.traffics[1:],
                utilizations=self.utilizations[1:])

            neighbors = []
            for nr_ways_first in range(nr_ways_min, nr_ways_max + 1):
                e_opt = e.optimal_calloc(nr_ways - nr_ways_first, policy)
                cbms = [ways_to_cbm(nr_ways_first)]
                nr_ways_used = nr_ways_first
                for cbm in e_opt.cbms:
                    nr_ways_this = cbm_to_ways(cbm)
                    cbms.append(
                        ways_to_cbm(nr_ways_this, nr_ways_used=nr_ways_used))
                    nr_ways_used += nr_ways_this
                neighbors.append(self.copy(cbms=tuple(cbms)))

            return neighbors

        # this is the max pps for incomplete cbms
        # note that this is definitely an overestimate
        def heuristic(nr_ways, policy):
            return objective_func(policy, self.pps_solos_full_cbm)

        def objective_func(policy, l):
            def eval_func(e):
                v = 100 - e.pps_predict_cat_normalized_full_cbm
                if policy == 'minmax':
                    return max(v)
                elif policy == 'minsum':
                    return sum(v)

            return min(l, key=eval_func)

        if policy == 'equal':
            assert (nr_ways % self.nr_pipelets == 0)
            cbms = []
            nr_ways_used = 0
            for i in range(self.nr_pipelets):
                nr_ways_this = int(nr_ways / self.nr_pipelets)
                cbms.append(
                    ways_to_cbm(nr_ways_this, nr_ways_used=nr_ways_used))
                nr_ways_used += nr_ways_this
            return self.copy(cbms=tuple(cbms))

        # base case
        if self.nr_pipelets == 1:
            return self.copy(cbms=(ways_to_cbm(nr_ways), ))

        return objective_func(policy, neighbors())

    def rtt_to_utilization(self, rtt):
        u_list = []
        all_cbms = \
            filter(lambda e: e.is_done,
                   [self.copy(cbms=tuple([ways_to_cbm(nr_ways)] + list(self.cbms[1:])))
                    for nr_ways in [config.cat['nr_ways_min'],
                                    config.cat['nr_ways_max']]])

        for e in all_cbms:
            e_u = e.all_utils
            if len(e_u) <= 1:
                continue
            rtt_l = np.array([e.rtt_95[0] for e in e_u])
            e_u_select = e_u[rtt_l < rtt]
            if len(e_u_select) == 0:
                print(rtt_l)
                return None
            import heapq
            u = min(heapq.nlargest(1, [e.utilizations[0] for e in e_u_select]))
            #if u == 98:
            #    u = 95
            #if e.pipelets[0] == 'mon':
            #    u = 90
            u_list.append(u)
        if len(u_list) == 0:
            return None
        return min(u_list)

    def selector(self, best=None, is_done=None):
        if best:
            r = self.best_run
            return np.array([r] if r else [])

        if is_done is None:
            return np.array(
                [self.copy(run_number=r) for r in config.run_numbers])
        elif is_done:
            return self.all_runs
        else:
            return self.all_runs_pending

    def start(self, retry=0, ignore=False, force=False):
        def _format_list(fmt, l):
            return ', '.join([fmt % i for i in l])

        if retry < 0:
            return False

        if not force and self.is_done:
            return True

        if not self.is_necessary:
            return True

        log_info(self.__str__())

        self.results = {}
        if not RuntimeManager().start(self):
            return self.start(retry=retry - 1, ignore=ignore, force=force)

        if self.has_results:
            MB = 1024 * 1024
            log_info('rx   (Mpps): %s' % _format_list('%6.3f', self.mpps))
            log_info(
                'tx   (Mpps): %s' % _format_list('%6.3f', self.tx_mpps_mean))
            log_info(
                'rx   (Mbps): %s' % _format_list('%6d', self.rx_mbps_mean))
            log_info(
                'tx   (Mbps): %s' % _format_list('%6d', self.tx_mbps_mean))
            log_info('rtt_95 (us): %s' % _format_list('%6d', self.rtt_95))
            log_info(
                'l3mr    (%%): %s' % _format_list('%6.2f', self.l3missrate))
            #log_info('l3oc   (MB): %s' % _format_list('%6.2f', self.llc_occupancy / MB))
            #log_info('mem  (MBps): %s' % _format_list('%6d', self.local_bytes / MB))
            log_info('mem  (MBps): r=%6d w=%6d' %
                     (self.llc_misses__mem_read / MB,
                      self.llc_misses__mem_write / MB))
            log_info('pcie (MBps): r=%6d w=%6d' %
                     (self.llc_references__pcie_read / MB,
                      self.llc_references__pcie_write / MB))
            log_info('ddiomr  (%%): r=%6.2f w=%6.2f' %
                     (100 * self.llc_misses__pcie_read / self.
                      llc_references__pcie_read,
                      100 * self.llc_misses__pcie_write / self.
                      llc_references__pcie_write))
            # log_info('all        :\n%s' % pformat(self.results, width=250))

        # is_done, fail_msgs = self.is_sane()
        is_done, fail_msgs = (True, ('',)) if self.has_results else (False, ('no results', ))

        if not is_done:
            log_error('Run sanity check failed')
            log_error(fail_msgs)
            log_debug(pformat(self.results, width=250))
            return self.start(retry=retry - 1, ignore=ignore, force=force)

        if not ignore:
            db.session.merge(self)
            flag_modified(self, 'results')
            db.session.flush()
            db.session.commit()

        return True

    # FIXME
    def stop(self):
        if hasattr(self, '_instances'):
            for i in self._instances:
                if i:
                    i.stop()


class RuntimeManager(object, metaclass=Singleton):
    def __init__(self):
        from resq.nf import NFInstanceManager
        from resq.port import PortManager
        self._loop = asyncio.get_event_loop()
        #_loop.set_debug(True)
        self.nfi_manager = NFInstanceManager()
        self.port_manager = PortManager()
        local_ports = self.port_manager.list(
            numa_node=config.run_socket_id, ring=False)
        self.tgen = MelvinGen(local_ports=local_ports)

        # clean up after a previous run
        cmds = [#['pkill', '-9', '-f', 'userlevel/click'],
                ['service', 'lldpd', 'stop'],
                ['service', 'irqbalance', 'stop']]
        for cmd in cmds:
            call(cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)

        disable_c_state()
        disable_p_state()
        disable_nmi_watchdog()
        disable_core_frequency_scaling()
        disable_uncore_frequency_scaling()
        configure_ddio()
        configure_irq_affinity()
        configure_pcie()
        init_dpdk()
        init_hugepages()
        init_lxc()
        init_netmap()
        init_rdt()

    @asyncio.coroutine
    def _monitor_pmu(self,
                     run,
                     warmup_sec=config.warmup_sec,
                     duration_sec=config.duration_sec):
        # FIXME: assumes NF has exactly one core
        # revert to the old model of one perf per NF
        cmd = {}
        proc = {}
        out = {}
        cmd['uncore'] = [
            'perf', 'stat', '--field-separator', ',', '--delay',
            str(warmup_sec), '--repeat',
            str(duration_sec - warmup_sec), '--event',
            ','.join(config.perf_events['uncore']), '--per-socket', '-a',
            'sleep', '1'
        ]
        # FIXME: pcm-memory takes 1s to start
        #cmd['membw'] = [config.pcm_memory_binary, '-csv', '--', 'sleep',
        #                str(duration_sec - 1)]
        # FIXME: pcm-pcie takes 1.36s to start
        #cmd['pcibw'] = [config.pcm_pcie_binary, '-csv', '-B', '--', 'sleep',
        #                str(duration_sec - 1.37)]

        nf_results = []
        for inst in self.nfi_manager.list():
            fut = asyncio.ensure_future(inst.monitor_pmu_async())
            nf_results.append(fut)
        for t in cmd.keys():
            proc[t] = asyncio.ensure_future(
                asyncio.
                create_subprocess_exec(    #'cgexec', '--sticky', '-g', 'cpuset:/',
                    *cmd[t],
                    stdin=DEVNULL,
                    stdout=PIPE,
                    stderr=PIPE))
        l3_flush()

        try:
            yield from asyncio.sleep(warmup_sec)

            nf_results = yield from asyncio.gather(*nf_results)
            nf_results = {
                k: [r[k] for r in nf_results]
                for k in nf_results[0].keys()
            }
            run.results.update(nf_results)

            proc['uncore'] = yield from proc['uncore']
            out = yield from proc['uncore'].stderr.read()
            for line in out.decode().splitlines():
                socket, _, value, unit, name, *etc = line.split(',')
                if socket != 'S%d' % config.run_socket_id:
                    continue
                name = re.sub(r'^[^/]*/', '', name).\
                    replace(':pp', '').\
                    replace('/', '').\
                    replace('-', '_').\
                    replace('.', '__')

                if name not in run.results:
                    run.results[name] = 0
                run.results[name] += int(float(value))

            # FIXME: mbps and bps here are bytes/sec
            if 'membw' in proc:
                proc['membw'] = yield from proc['membw']
                out = yield from proc['membw'].stdout.read()
                lines = [l.split(';') for l in out.decode().splitlines()]
                assert (len(lines) == 3)
                index_skt = lines[0].index('SKT%d' % config.run_socket_id)
                index_r = lines[1].index('Mem Read (MB/s)', index_skt)
                index_w = lines[1].index('Mem Write (MB/s)', index_skt)
                run.results['mem_read_mbps'] = int(float(lines[2][index_r]))
                run.results['mem_write_mbps'] = int(float(lines[2][index_w]))

            if 'pcibw' in proc:
                proc['pcibw'] = yield from proc['pcibw']
                out = yield from proc['pcibw'].stdout.read()
                lines = [l.split(',') for l in out.decode().splitlines()]
                assert (len(lines) == 3 and int(lines[config.run_socket_id + 1]
                                                [0]) == config.run_socket_id)
                for name, value in zip(lines[0][1:], lines[config.run_socket_id
                                                           + 1][1:]):
                    name = name.replace(' ', '_') \
                        .replace('(B)', 'mbps') \
                        .replace('Rd', 'Read') \
                        .replace('Wr', 'Write') \
                        .lower()
                    value = int(value)
                    if 'mbps' in name:
                        value = value / 1000000.
                    run.results[name] = value
        except asyncio.CancelledError:
            log_error('monitor_pmu failed: Timeout')
        except Exception as e:
            log_error('monitor_pmu failed: %s' % e)

        for p in proc.values():
            exitcode = yield from p.wait()
            assert (exitcode == 0)

    @asyncio.coroutine
    def _monitor_traffic(self, run):
        # FIXME: doesn't work for MLC
        try:
            tx_mbps_l = run.tx_mbps_request
            ports = [
                i.ports[0].name if i.nr_ports > 0 else None
                for i in self.nfi_manager.list()
            ]
            randomize_payloads = [
                NF(nf).randomize_payload for nf in run.pipelets
            ]

            r = yield from \
                self.tgen.measure(peer_ports=ports,
                                  tx_mbps_l=tx_mbps_l,
                                  traffics=run.traffics,
                                  randomize_payloads=randomize_payloads)

            if r:
                # convert a list of dicts to a dict of lists
                keys = sorted(list(set([k for i in r for k in i.keys()])))
                r = {k: [d[k] if k in d else -1 for d in r] for k in keys}
                run.results.update(r)
        except asyncio.CancelledError:
            log_error('monitor_traffic failed: Timeout')
        except Exception as e:
            log_error('monitor_traffic failed: %s' % e)

    def monitor(self, run):
        subtasks = [self._monitor_pmu(run), self._monitor_traffic(run)]
        timeout = config.duration_sec + 1.5

        done, pending = self._loop.run_until_complete(
            asyncio.wait(subtasks, timeout=timeout))
        for i in pending:
            if not i.cancelled():
                i.cancel()
        if len(pending) > 0:
            return False

        if 'cache_references' not in run.results:
            log_error('Failed to collect PMU results')
            return False
        if 'rx_mbps_mean' not in run.results:
            log_error('Failed to collect traffic results')
            self.tgen.stop()
            return False

        for instance in self.nfi_manager.list():
            instance.measure_count += 1

        count = np.array([
            instance.measure_count > instance.measure_count_min
            for instance in self.nfi_manager.list()
        ])
        if not np.all(count):
            res = self.monitor(run)
            return res

        return True

    def start(self, run):
        if any(
                np.logical_and(run.tx_mbps_request <= 0, run.utilizations !=
                               100)):
            log_error('Insufficient info to run this experiment.')
            return False

        if self.port_manager.status != NFStatus.ready:
            log_info('Starting the port manager (BESS).')
            self.port_manager.start()
        if self.tgen.status != NFStatus.ready:
            log_info('Starting the traffic generator (MelvinGen).')
            self.tgen.start()

        if not self.port_manager.wait(timeout=15):
            log_error('Failed to boot the port manager.')
            return False
        if not self.tgen.wait(timeout=15):
            log_error('Failed to boot the traffic generator')
            return False

        if not self.nfi_manager.create_instances(run):
            log_error('Failed to create instances')
            return False

        log_debug('Starting instancess')
        self.nfi_manager.start()
        if not self.nfi_manager.wait(timeout=60):
            for i, inst in enumerate(self.nfi_manager.list()):
                if inst.status != NFStatus.ready:
                    log_error('Failed to boot pipelet #%d: %s' % ((i + 1),
                                                                  inst.name))
                    inst.stop()
            return False

        log_debug('Enforcing limits')
        if not self.nfi_manager.enforce_limits():
            log_error('Failed to enforce limits.')
            return False

        log_debug('Monitoring the run')
        return self.monitor(run)

    def stop(self):
        self.port_manager.stop()
        self.nfi_manager.stop()
        self.tgen.stop()
        self._loop.close()


atexit.register(RuntimeManager().stop)
