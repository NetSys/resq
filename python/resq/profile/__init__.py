#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

import resq.config as config
from resq.nf import NFManager
from resq.run import Run
from resq.util.log import log_error, log_info
from resq.util import ways_to_cbm


class Profile(object):
    def __init__(self, pipelet, traffic):
        self.pipelet = pipelet
        self.traffic = traffic

    def _runs(self, cbms=None, utilizations=None, best=None, is_done=None):
        if not cbms:
            nr_ways_min = config.cat['nr_ways_min']
            nr_ways_max = config.cat['nr_ways_max']
            cbms = [ways_to_cbm(n)
                    for n in range(nr_ways_min, nr_ways_max + 1)]
        if not utilizations:
            utilizations = config.utilizations

        l = []
        for utilization in utilizations:
            for cbm in cbms:

                l += list(Run(pipelets=[self.pipelet],
                              cbms=[cbm],
                              traffics=[self.traffic],
                              utilizations=[utilization]).selector(
                                  best=best, is_done=is_done))

        return l

    def l3_xput_approx(self, utilization=100):
        """
        returns a piecewise linear fit of the cache curve
        (coeffs, nr_ways_pivot, nr_ways_max)
        """

        def fit(x, y):
            coeff = np.polyfit(x, y, deg=1)
            yfit = np.polyval(coeff, x)
            e = np.abs(yfit - y) / y
            return (e, coeff)

        try:
            # FIXME: not the best run
            runs = self.runs_cache(utilizations=[utilization], best=True)

            d = np.array(
                [(r.l3_ways[0], r.pps_normalized_full_cbm[0]) for r in runs
                 if r.pps_normalized_full_cbm[0] <= 98 or r.cbms[0] in [1, 3]])
            d.sort(axis=0)
            x, y = d.T
            xdiff = np.diff(x)
            if set(xdiff) != set([1]):
                assert(len(set(xdiff)) == 2)
                idx = np.argmax(xdiff)
                x, y = x[:idx], y[:idx]
            assert(x[0] == 1 and x[-1] <= 20)

            err, coeff = fit(x, y)
            coeff[0] = max(coeff[0], 0)
            if (err < 0.01).all():
                return ((coeff, coeff), x[-1], x[-1])
            else:
                err = {}
                for pivot in range(1, len(x)):
                    err1, coeff1 = fit(x[:pivot + 1], y[:pivot + 1])
                    err2, coeff2 = fit(x[pivot:], y[pivot:])
                    err[pivot] = (max(
                        max(err1), max(err2)), (
                            (coeff1, coeff2), x[pivot], x[-1]))
                    assert(coeff1[0] >= 0 and coeff2[0] >= 0)
                pivot = min(err, key=lambda k: err[k][0])
                return err[pivot][1]
        except Exception as e:
            import sys
            log_error(e)
            log_error('line {}'.format(sys.exc_info()[-1].tb_lineno))
        return False

    def find(self, cbms=None, utilizations=None):
        return Run(pipelets=[self.pipelet],
                   cbms=cbms,
                   traffics=[self.traffic],
                   utilizations=utilizations)

    def runs_cache(self, utilizations=[100], best=None, is_done=None):
        return self._runs(utilizations=utilizations,
                          best=best,
                          is_done=is_done)

    def runs_latency(self,
                     cbms=[config.cat['cbm_min'], config.cat['cbm_max']],
                     best=None,
                     is_done=None):
        return self._runs(cbms=cbms, best=best, is_done=is_done)

    def runs_chain(self):
        nr_ways_min = config.cat['nr_ways_min']
        nr_ways_max = config.cat['nr_ways_max']
        chain_length_max = nr_ways_max / nr_ways_min
        cbms = [ways_to_cbm(nr_ways_min, nr_ways_min * i)
                for i in range(chain_length_max)]
        l = [Run(pipelets=[[self.pipelet] * chain_length],
                 cbms=[cbms[0:chain_length]],
                 traffics=[self.traffic],
                 utilizations=[100],
                 run_number=run_number)
             for run_number in config.run_numbers
             for traffic in config.traffics.keys()
             for chain_length in range(2, chain_length_max + 1, 2)]
        return l

    def runs_l3(self,
                nr_pipelets=12,
                nr_ways=12,
                other_nf_types=['mlc', 'real', 'syn'],
                utilizations=[100],
                isolated=False,
                best=None,
                is_done=None):
        assert(all(mix in set(['mlc', 'real', 'syn'])
                   for mix in other_nf_types))

        cbms = [ways_to_cbm(nr_ways)] * nr_pipelets
        if isolated:
            cbms = [config.cat['cbm_min']] + \
                [ways_to_cbm(nr_ways - config.cat['nr_ways_min'],
                             config.cat['nr_ways_min'])] * (nr_pipelets - 1)

        l = []
        for mix in other_nf_types:
            for other_nf in NFManager().list(real='real' == mix,
                                           mlc='mlc' == mix,
                                           syn='syn' == mix):
                #utilizations = [100] #config.utilizations if mix != 'mlc' else [100]
                for utilization in utilizations:
                    l += list(Run(pipelets=[self.pipelet] + [other_nf] * (
                        nr_pipelets - 1),
                                  cbms=cbms,
                                  traffics=[self.traffic] * nr_pipelets,
                                  utilizations=[utilization] + [100] *
                                  (nr_pipelets - 1)).selector(best=best,
                                                              is_done=is_done))

        return l

    def do(self, force=False):
        def key_(e):
            return (list(e.pipelets[1:]), e.pipelets[0], e.cbms[0],
                    e.utilizations[0] != 100)

        s = []
        s += self.runs_cache()
        if self.pipelet not in ['ipsec', 'wanopt']:
            s += self.runs_latency()
        s += self.runs_l3(nr_pipelets=12,
                          nr_ways=12,
                          other_nf_types=['real', 'syn'],
                          utilizations=[100],
                          isolated=False)
        s += self.runs_l3(nr_pipelets=12,
                          nr_ways=12,
                          other_nf_types=['real', 'syn'],
                          utilizations=[100],
                          isolated=True)
        s = sorted(s, key=key_)

        failures_max = 20
        failures = 0
        while len(s) > 0 and failures < failures_max:
            e = s.pop(0)
            if not e.is_necessary and e.is_done:
                from resq.util import db
                db.session.delete(e)
                db.session.commit()
            if not e.start(force=force, retry=1):
                failures += 1

            log_info('Remaining: %d' % len(s))

        if failures >= failures_max:
            log_error('More than %d failures... aborting!' % failures)
            return False

        return True
