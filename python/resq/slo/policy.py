#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import glob
from gurobipy import *
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import array
from pprint import pprint

from resq.profile import Profile
from resq.run import Run
from resq.nf import NFManager


class Policy:
    @staticmethod
    def evaluate(nr_cores_per_node=8):
        result = []
        for fout in glob.glob('sla/core-%d/*.out' % nr_cores_per_node):
            fin = fout.replace('out', 'in')
            with open(fin) as f:
                terms = eval(f.read())
            with open(fout) as f:
                sol = eval(f.read())

            methods = {
                'offline.': 'CAT-Offline',
                'online_cat.': 'CAT-Online',
                'online_e2_cat.': 'CAT-Equal',
                'online_predict_': 'Predict',
                'online_e2.': 'E2'
            }

            method = None
            for key in methods:
                if key in fout:
                    method = methods[key]
                    break

            categories = {
                'all': 'All',
                'insensitive': 'Insensitive',
                'sensitive': 'Sensitive',
            }

            category = None
            for key in categories:
                if fout.startswith('sla/core-%d/%s' %
                                   (nr_cores_per_node, key)):
                    category = categories[key]

            print('Evaluating E2 violations for %s' % fout)
            violations = [0 for t in terms]
            for node in sol:
                nf_list = [terms[idx]['app'] for (idx, cache) in node]
                print(nf_list)
                run = Run(nf_list)
                if not run.is_done:
                    print('WARNING: do not have the results for this...')
                    continue
                num = len(nf_list)
                pps = run.pps_normalized_full_cbm
                rtt = rtt_95_normalized_full_cbm
                for i in xrange(num):
                    term_idx = node[i][0]
                    if pps[i] < terms[idx]['xput'] or (
                            not rtt[i].isnan() and rtt[i] > terms[idx]['rtt']):
                        violations[term_idx] = 1
            print('Violations: %d/%d' % (sum(violations), len(violations)))
            result.append((category, method, len(sol)))

        pprint(sorted(result))
        '''
        width = 0.25
        pos = list(range(3))
        fig, ax = plt.subplots(figsize=(10,5))
        plt.bar(pos, )
        '''
        '''
        def xput(term, cache):
            pivot = slos[term]['pivot']
            coeff = slos[term]['coeff']
            util = slos[term]['util']
            util = util if util else 1
            max_xput = 0
            if cache <= pivot:
                max_xput = coeff[0][0] * cache + coeff[0][1]
            else:
                max_xput = coeff[1][0] * cache + coeff[1][1]
            return util * max_xput

        for fout in glob.glob('sla/*.out'):
            name = fout.replace('.out', '')
            with open(fout) as f:
                sol = eval(f.read())
            with open(fout.replace('.out', '.in')) as f:
                slos = eval(f.read())
            terms = range(len(slos))
            term_pps = [0] * len(terms)
            target_pps = [slo['xput'] for slo in slos]
            nr_cores = sum([len(node) for node in sol])
            nr_nodes = len(sol)
            for node in sol:
                term_idx = [i[0] for i in node]
                apps = [slos[term]['app'] for term in term_idx]
                utils = [slos[term]['util'] for term in term_idx]
                e = None
                print(sorted(zip(apps, utils)))
                continue
                for i, term in enumerate(term_idx):
                    pps_mean = np.mean(it.compress(e.pps, e.apps == app))
                    term_pps[term] += e.pps_normalized_full_cbm[i]
            continue
            nr_violations = 0
            dist = []
            for i, (observed, target) in enumerate(zip(term_pps, target_pps)):
                if observed < target:
                    e = 100 * (target - observed) / target
                    dist.append(e)
                    # print('%s: slo %d violated by %f percent' % (name, i, e))
                    nr_violations += 1
            violations = {}
            for n in [2, 5, 10, 15, 20, 30, 40, 50]:
                nr_violations = len([i for i in dist if i > n])
                violations[n] = 100 * nr_violations / nr_slos

            nr_instances_lb = [int(math.ceil(slos[term]['xput'] / xput(term, max_cache[app])))
                               for term in terms]
            nr_instances_ub = [int(math.ceil(slos[term]['xput'] / xput(term, 1)))
                               for term in terms]
            nr_nodes_lb = int(math.ceil(sum(nr_instances_lb) / float(nr_cores_per_node)))

            print('%s & %d & %d & %d & %d & %2.2f & %2.2f & %2.2f & %2.2f & %2.2f\\\\' % (name, nr_slos, nr_cores, nr_nodes, nr_nodes_lb, violations[2], violations[5], violations[10], violations[15], violations[20]))
        '''

    @staticmethod
    def generate_slos(rtt=100, xput=90, nr_slos=60):
        apps = {
            'sensitive': ['mon', 'ip_131k', 'mazunat', 'snort'],
            'insensitive': ['efficuts_32k', 'firewall_250', 'ipsec'],
            'neutral': ['suricata', 'wanopt']
        }
        slos = {
            'insensitive-%d-rtt%d-xput%d' % (nr_slos, rtt, xput):
            apps['insensitive'] * int(nr_slos / len(apps['insensitive'])),
            'sensitive-%d-rtt%d-xput%d' % (nr_slos, rtt, xput):
            apps['sensitive'] * int(nr_slos / len(apps['sensitive'])),
            'all-%d-rtt%d-xput%d' % (nr_slos, rtt, xput):
            [app for l in apps.values() for app in l] *
            int(nr_slos / sum([len(a) for a in apps.values()]))
        }
        for type_ in slos.keys():
            d = []
            for app in slos[type_]:
                profile = Profile(pipelet=app, traffic='u100k_60')
                coeff, pivot, max_cache = profile.l3_xput_approx()
                util = profile.find().rtt_to_utilization(rtt)
                d.append({
                    'app': app,
                    'coeff': coeff,
                    'pivot': pivot,
                    'util': util / 100 if util else 1,
                    'max_cache': max_cache,
                    'rtt': rtt,
                    'xput': xput
                })
                print(app, util)
            slos[type_] = d
        return slos

    @staticmethod
    def online_predict_nsdi12(slos, nr_cores_per_node=16, xput=90):
        def xput_min(app):
            return slos[app]['util'] * (
                slos[app]['coeff'][0][0] * 1 + slos[app]['coeff'][0][1])

        def xput_max(app):
            return slos[app]['util'] * (
                slos[app]['coeff'][1][0] * slos[app]['max_cache'] +
                slos[app]['coeff'][1][1])

        terms = []
        for app in slos:
            app_name = app['app']
            profile = Profile(pipelet=app_name, traffic='u100k_60')
            app_cache = profile.find().min_l3ways(app['xput'])
            #app_num = int(math.ceil(app['xput'] / xput_max(app)))
            terms.append((app_name, app_cache))

        terms.sort(key=lambda t: t[1], reverse=True)
        nodes = []
        i = 0
        for name, cache in terms:
            done = False
            for node in nodes:
                if len(node) >= nr_cores_per_node:
                    continue
                pipelets = [slos[idx]['app']
                            for idx, app_cache in node] + [name]
                traffics = ['u100k_60' for _ in pipelets]
                #print(pipelets)
                #print(traffics)
                run = Run(pipelets=pipelets, traffics=traffics)
                new_pps = run.pps_predict_nsdi12_normalized_full_cbm
                if (np.array(new_pps) > xput).all():
                    done = True
                    node.append((i, cache))
                    break
            if not done:
                nodes.append([(i, cache)])
            i += 1
        return nodes

    '''
    @staticmethod
    def online_cat(slos, nr_cores_per_node=16, nr_ways=18):
        def xput_min(app):
            return slos[app]['util'] * (slos[app]['coeff'][0][0] * 1 + slos[app]['coeff'][0][1])

        def xput_max(app):
            return slos[app]['util'] * (slos[app]['coeff'][1][0] * slos[app]['max_cache'] + slos[app]['coeff'][1][1])

        terms = []
        for app in slos:
            app_name = app['app']
            profile = Profile(pipelet=app_name, traffic='u100k_60')
            app_cache = profile.find().min_l3ways(app['xput'])
            #app_num = int(math.ceil(app['xput'] / xput_max(app)))
            terms.append((app_name, app_cache))

        terms.sort(key=lambda t: t[1], reverse=True)
        nodes = []
        i = 0
        for name, cache in terms:
            done = False
            for node in nodes:
                remaining_cores = nr_cores_per_node - len(node)
                remaining_cache = nr_ways - sum([core[1] for core in node])
                if remaining_cores >= 1 and remaining_cache >= cache:
                    node.append((i, cache))
                    done = True
                    break
            if not done:
                nodes.append([(i, cache)])
            i += 1
        return nodes
    '''

    @staticmethod
    def online_cat_binpack(slos, nr_cores_per_node=16, nr_ways=18):
        def xput(slo, cache):
            pivot = slo['pivot']
            if cache < pivot:
                return slo['util'] * (
                    slo['coeff'][0][0] * cache + slo['coeff'][0][1])
            else:
                return slo['util'] * (
                    slo['coeff'][1][0] * cache + slo['coeff'][1][1])

        terms = []
        for i, slo in enumerate(slos):
            profile = Profile(pipelet=slo['app'], traffic='u100k_60')

            target_xput = slo['xput']
            cache_line = nr_ways / nr_cores_per_node
            K = int(math.ceil(target_xput / xput(slo, cache_line)))

            caches = [1] * K
            xputs = [xput(slo, c) for c in caches]

            done = sum(xputs) > target_xput
            while not done:
                for i in range(K):
                    caches[i] += 1
                    xputs[i] = xput(slo, caches[i])
                    if sum(xputs) > target_xput:
                        done = True
                        break

            terms.append((i, caches))

        nodes = []
        for idx, caches in terms:
            num = len(caches)
            while num > 0:
                done = False
                for node in nodes:
                    used_cache = sum([c for _, c in node])
                    if len(node) < nr_cores_per_node and used_cache + caches[
                            num - 1] <= nr_ways:
                        node.append((idx, caches[num - 1]))
                        num -= 1
                        done = True
                        break
                if not done:
                    nodes.append([(idx, caches[num - 1])])
                    num -= 1
        return nodes

    @staticmethod
    def online_cat_greedy(slos, nr_cores_per_node=16, nr_ways=18):
        def xput(app, cache):
            pivot = slos[app]['pivot']
            if cache < pivot:
                return slos[app]['util'] * (slos[app]['coeff'][0][0] * cache +
                                            slos[app]['coeff'][0][1])
            else:
                return slos[app]['util'] * (slos[app]['coeff'][1][0] * cache +
                                            slos[app]['coeff'][1][1])

        terms = []
        for app in slos:
            app_name = app['app']
            profile = Profile(pipelet=app_name, traffic='u100k_60')
            app_xput = app['xput']
            terms.append((app_name, app_xput))

        terms.sort(key=lambda t: t[1], reverse=True)
        nodes = [[]]
        i = 0
        for name, app_xput in terms:
            profile = Profile(pipelet=app_name, traffic='u100k_60')
            cache_line = nr_ways / nr_cores_per_node
            num = int(math.ceil(app_xput / xput(i, cache_line)))
            while num > 0:
                if len(nodes[-1]) < nr_cores_per_node:
                    nodes[-1].append((i, cache_line))
                else:
                    nodes.append([(i, cache_line)])
                num -= 1
            i += 1
        return nodes

    @staticmethod
    def online_e2(slos, nr_cores_per_node=16):
        def xput_min(app):
            return slos[app]['util'] * (
                slos[app]['coeff'][0][0] * 1 + slos[app]['coeff'][0][1])

        def xput_max(app):
            return slos[app]['util'] * (
                slos[app]['coeff'][1][0] * slos[app]['max_cache'] +
                slos[app]['coeff'][1][1])

        terms = []
        for app in slos:
            app_name = app['app']
            profile = Profile(pipelet=app_name, traffic='u100k_60')
            app_cache = profile.find().min_l3ways(app['xput'])
            terms.append((app_name, app_cache))

        terms.sort(key=lambda t: t[1], reverse=True)
        nodes = [[]]
        i = 0
        for name, cache in terms:
            if len(nodes[-1]) < nr_cores_per_node:
                nodes[-1].append((i, cache))
            else:
                nodes.append([(i, cache)])
            i += 1
        return nodes

    @staticmethod
    def offline(slos, nr_cores_per_node=16, cache_size=18):
        def terminator(m, where):
            if where == GRB.Callback.MIP:
                time = m.cbGet(GRB.Callback.RUNTIME)
                best = m.cbGet(GRB.Callback.MIP_OBJBST)

                if time > 60 and best < GRB.INFINITY:
                    # nodecnt = m.cbGet(GRB.callback.MIP_NODCNT)
                    # solcnt = m.cbGet(GRB.callback.MIP_SOLCNT)
                    objbst = m.cbGet(GRB.callback.MIP_OBJBST)
                    objbnd = m.cbGet(GRB.callback.MIP_OBJBND)

                    if abs(objbst - objbnd) < 0.10 * (1.0 + abs(objbst)):
                        m.terminate()
                    elif time > 1200 and abs(objbst - objbnd) < 0.30 * (
                            1.0 + abs(objbst)):
                        m.terminate()

        def xput_min(term):
            return slos[term]['util'] * (
                slos[term]['coeff'][0][0] + slos[term]['coeff'][0][1])

        def xput_max(term):
            return slos[term]['util'] * (
                slos[term]['coeff'][1][0] * slos[term]['max_cache'] +
                slos[term]['coeff'][1][1])

        def xput(term, instance):
            # C_{ij}
            cache_var = cache[term, instance]

            # for all m, lambda_{ijm}
            coeff_sel = [coeff_select[term, instance, i] for i in range(2)]

            coeff = slos[term]['coeff']
            util = slos[term]['util']
            util = util if util else 1.0

            max_xput = coeff_sel[0] * (coeff[0][0] * cache_var + coeff[0][1]) + \
                coeff_sel[1] * (coeff[1][0] * cache_var + coeff[1][1])
            print(term, instance, util, max_xput)
            return (util - 0.03) * max_xput

        m = Model("policy2a")

        terms = range(len(slos))
        nr_instances_lb = [int(math.ceil(slos[term]['xput'] / xput_max(term)))
                           for term in terms]
        nr_instances_ub = [int(math.ceil(slos[term]['xput'] / xput_min(term)))
                           for term in terms]
        nr_nodes_lb = int(math.ceil(sum(nr_instances_lb) / nr_cores_per_node))
        nr_nodes_ub = int(math.ceil(sum(nr_instances_ub) / nr_cores_per_node))
        print('nodes lb: %d, ub: %d' % (nr_nodes_lb, nr_nodes_ub))

        nodes = range(nr_nodes_ub)
        instances = [list(range(i)) for i in nr_instances_ub]

        # lambda_{ijm}
        coeff_select = {(term, instance, i):
                        m.addVar(vtype=GRB.BINARY,
                                 name='CSEL_%d_%d_%d' % (term, instance, i))
                        for term in terms for instance in instances[term]
                        for i in range(2)}

        # C_{ij}
        cache = {(term, instance):
                 m.addVar(lb=0,
                          ub=slos[term]['max_cache'], vtype=GRB.INTEGER,
                          name='C_%d_%d' % (term, instance))
                 for term in terms for instance in instances[term]}

        # I_{ijk}
        affinity = {(term, instance, node):
                    m.addVar(vtype=GRB.BINARY,
                             name='I_%d_%d_%d' % (term, instance, node))
                    for term in terms for instance in instances[term]
                    for node in nodes}

        # N_k
        node_active = {node: m.addVar(vtype=GRB.BINARY,
                                      name='N_%d' % (node))
                       for node in nodes}

        # Integrate new variables
        m.update()

        # forall k, \sum_{i,j} I_{ijk}
        node_cores = {node: quicksum(affinity[term, instance, node]
                                     for term in terms
                                     for instance in instances[term])
                      for node in nodes}

        # \sum_{i,j} C_{ij} I_{ijk}
        node_cache = {
            node:
            quicksum(cache[term, instance] * affinity[term, instance, node]
                     for term in terms for instance in instances[term])
            for node in nodes
        }

        # forall i, \sum_{jk} I_{ijk}
        nr_instances = {term: quicksum(affinity[term, instance, node]
                                       for node in nodes
                                       for instance in instances[term])
                        for term in terms}

        # forall i,j, \sum_{k} I_{ijk}
        instance_active = {(term, instance):
                           quicksum(affinity[term, instance, node]
                                    for node in nodes)
                           for term in terms for instance in instances[term]}

        # forall i, \sum_j T_{ij}
        term_xput = {term: quicksum(xput(term, instance)
                                    for instance in instances[term])
                     for term in terms}

        # \sum_k N_k
        nr_nodes_active = quicksum(node_active.values())

        m.setObjective(nr_nodes_active, GRB.MINIMIZE)

        #m.addConstr(nr_nodes_active >= nr_nodes_lb, name='nr_nodes_active_lb')

        for term in terms:
            # target throughput constraint
            m.addConstr(term_xput[term] >= slos[term]['xput'],
                        name='term_xput_%d' % term)
            # bounds on number of instances just to speed things up
            m.addConstr(nr_instances[term] >= nr_instances_lb[term],
                        name='nr_instances_lb_%d' % term)
            m.addConstr(nr_instances[term] <= nr_instances_ub[term],
                        name='nr_instances_ub_%d' % term)
            for inst in instances[term]:
                # each instance is active on at most one node
                m.addSOS(GRB.SOS_TYPE1,
                         [affinity[term, inst, node] for node in nodes])
                # if instance_active one of the coeff_select's are 1, otherwise both are 0
                m.addConstr(
                    coeff_select[term, inst, 0] + coeff_select[term, inst, 1] -
                    instance_active[term, inst] == 0,
                    name='sum_coeff_select_%d_%d' % (term, inst))
                # following conditions for piecewise linear model of xput/cache curve
                # if 1 < cache < pivot then first approx is used otherwise (pivot < cache < max) the second.
                m.addConstr(cache[term, inst] <= coeff_select[term, inst, 0] *
                            slos[term]['pivot'] + coeff_select[term, inst, 1] *
                            slos[term]['max_cache'], name='cache_ub_%d_%d' %
                            (term, inst))
                m.addConstr(cache[term, inst] >= coeff_select[term, inst, 1] *
                            slos[term]['pivot'] + coeff_select[term, inst, 0] *
                            1, name='cache_lb_%d_%d' % (term, inst))

        for node in nodes:
            # node is active iff at least one core on it is active
            m.addConstr(
                node_cores[node] - nr_cores_per_node * node_active[node] <= 0,
                name='node_active1_%d' % node)
            m.addConstr(node_cores[node] - node_active[node] >= 0,
                        name='node_active2_%d' % node)
            m.addConstr(node_cores[node] <= nr_cores_per_node,
                        name='node_cores_%d' % node)
            m.addConstr(node_cache[node] <= cache_size,
                        name='node_cache_%d' % node)

        #m.update()
        #m.write('sla/policy2a.lp')
        m.optimize(terminator)

        if m.status == GRB.status.OPTIMAL:
            print('Optimal objective: %g' % m.objVal)
        elif m.status == GRB.status.INF_OR_UNBD:
            print('m is infeasible or unbounded')
            return None
        elif m.status == GRB.status.INFEASIBLE:
            print('m is infeasible')
            return None
        elif m.status == GRB.status.UNBOUNDED:
            print('m is unbounded')
            return None
        else:
            print('Optimization ended with status %d' % m.status)

        for term in terms:
            x = term_xput[term].getValue()
            cores = sum(instance_active[term, inst].getValue()
                        for inst in instances[term])
            print('term %-15s: %d cores, %-5.3g >= %-5.3g' %
                  (term, cores, x, slos[term]['xput']))

        result = []
        for node in nodes:
            x = ['%d ways-%s-%d' %
                 (cache[term, inst].x, slos[term]['app'], inst)
                 for term in terms for inst in instances[term]
                 if affinity[term, inst, node].x > 0.5]
            r = [(term, cache[term, inst].x)
                 for term in terms for inst in instances[term]
                 if affinity[term, inst, node].x > 0.5]
            if x != []:
                print('node %d: %s' % (node, x))
                result.append(r)

        print('min_number_of_nodes: %g' % m.objVal)
        return result

    @staticmethod
    def run_online_predict_nsdi12(slos, desc, nr_cores_per_node):
        outfile = 'sla/%s-online_predict_nsdi12.out' % desc
        infile = 'sla/%s-online_predict_nsdi12.in' % desc
        start = time.time()
        r = Policy.online_predict_nsdi12(slos,
                                         nr_cores_per_node=nr_cores_per_node)
        print('** %s-online_predict_nsdi12 took: %d' %
              (desc, time.time() - start))
        if r:
            with open(outfile, 'w') as f:
                pprint(r, f)
            with open(infile, 'w') as f:
                pprint(slos, f)

    @staticmethod
    def run_online_e2(slos, desc, nr_cores_per_node):
        outfile = 'sla/%s-online_e2.out' % desc
        infile = 'sla/%s-online_e2.in' % desc
        start = time.time()
        r = Policy.online_e2(slos, nr_cores_per_node=nr_cores_per_node)
        print('** %s-online_e2 took: %d' % (desc, time.time() - start))
        if r:
            with open(outfile, 'w') as f:
                pprint(r, f)
            with open(infile, 'w') as f:
                pprint(slos, f)

    @staticmethod
    def run_online_cat_greedy(slos, desc, nr_cores_per_node):
        outfile = 'sla/%s-online_cat_greedy.out' % desc
        infile = 'sla/%s-online_cat_greedy.in' % desc
        start = time.time()
        r = Policy.online_cat_greedy(slos, nr_cores_per_node=nr_cores_per_node)
        print('** %s-online_cat_greedy took: %d' % (desc, time.time() - start))
        if r:
            with open(outfile, 'w') as f:
                pprint(r, f)
            with open(infile, 'w') as f:
                pprint(slos, f)

    @staticmethod
    def run_online_cat_binpack(slos, desc, nr_cores_per_node):
        outfile = 'sla/%s-online_cat_binpack.out' % desc
        infile = 'sla/%s-online_cat_binpack.in' % desc
        start = time.time()
        r = Policy.online_cat_binpack(slos,
                                      nr_cores_per_node=nr_cores_per_node)
        print('** %s-online_cat_binpack took: %d' %
              (desc, time.time() - start))
        if r:
            with open(outfile, 'w') as f:
                pprint(r, f)
            with open(infile, 'w') as f:
                pprint(slos, f)

    @staticmethod
    def run_offline(slos, desc, nr_cores_per_node):
        outfile = 'sla/%s-offline.out' % desc
        infile = 'sla/%s-offline.in' % desc
        start = time.time()
        r = Policy.offline(slos, nr_cores_per_node=nr_cores_per_node)
        print('** %s-offline took: %d' % (desc, time.time() - start))
        if r:
            with open(outfile, 'w') as f:
                pprint(r, f)
            with open(infile, 'w') as f:
                pprint(slos, f)

    @staticmethod
    def run():
        slos_all = Policy.generate_slos(nr_slos=200)
        nr_cores_per_node = 9
        for k, slos in slos_all.items():
            print("solving: %s" % k)
            Policy.run_offline(slos, k, nr_cores_per_node)
            Policy.run_online_e2(slos, k, nr_cores_per_node)
            Policy.run_online_cat_binpack(slos, k, nr_cores_per_node)
            Policy.run_online_cat_greedy(slos, k, nr_cores_per_node)
            Policy.run_online_predict_nsdi12(slos, k, nr_cores_per_node)
        #Policy.evaluate()
