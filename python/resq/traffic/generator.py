#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import struct

import resq.config as config
from resq.nf import NFInstance, NFStatus
from resq.port import Port
from resq.traffic import Traffic
from resq.util import Singleton
from resq.util.log import log_debug, log_error


class MelvinGen(NFInstance, metaclass=Singleton):
    def __init__(self, local_ports=[]):
        super().__init__(name='melvingen')
        self.args = ''
        for p in local_ports:
            if p.is_virtual:
                self.args += ' --vdev '
            else:
                self.args += ' -w '
            self.args += p.neighbor_pciaddr
        self.host = config.melvingen_host
        self.port = int(config.melvingen_port)
        self.reader = None
        self.writer = None

    @asyncio.coroutine
    def _recv_status(self):
        from resq.traffic.protobuf.status_pb2 import Status
        length = yield from self.reader.read(4)
        length = struct.unpack('>L', length)[0]
        buf = yield from self.reader.read(length)

        status = Status()
        status.ParseFromString(buf)

        if status.type == Status.SUCCESS:
            return True, None
        elif status.type == Status.STATS:
            stats = {}
            for ps in status.stats:
                #stats[ps.port] = {}
                #for field in ps.ListFields():
                #    stats[ps.port][field[0].name] = field[1]
                stats[ps.port] = {
                    'rtt_0': ps.rtt_0,
                    'rtt_100': ps.rtt_100,
                    'rtt_25': ps.rtt_25,
                    'rtt_50': ps.rtt_50,
                    'rtt_75': ps.rtt_75,
                    'rtt_90': ps.rtt_90,
                    'rtt_95': ps.rtt_95,
                    'rtt_99': ps.rtt_99,
                    'rtt_mean': ps.rtt_avg,
                    'rtt_samples': ps.n_rtt,
                    'rtt_std': ps.rtt_std,
                    'rx_mbps_mean': ps.avg_rxbps,
                    'rx_mbps_std': ps.std_rxbps,
                    'rx_mbps_wire_mean': ps.avg_rxwire,
                    'rx_mbps_wire_std': ps.std_rxwire,
                    'rx_mpps_mean': ps.avg_rxmpps,
                    'rx_mpps_std': ps.std_rxmpps,
                    'tx_mbps_mean': ps.avg_txbps,
                    'tx_mbps_std': ps.std_txbps,
                    'tx_mbps_wire_mean': ps.avg_txwire,
                    'tx_mbps_wire_std': ps.std_txwire,
                    'tx_mpps_mean': ps.avg_txmpps,
                    'tx_mpps_std': ps.std_txmpps
                }
            return True, stats

        # status.type == Status.FAIL
        return False, None

    @asyncio.coroutine
    def _send_jobs(self, jobs_dict):
        from resq.traffic.protobuf.job_pb2 import Job, Request
        jobs = []
        for job_dict in jobs_dict:
            job = Job()
            for k, v in job_dict.items():
                assert(hasattr(job, k))
                setattr(job, k, v)
            jobs.append(job)
        request = Request()
        request.jobs.extend(jobs)
        buf = request.SerializeToString()
        length = struct.pack('>L', len(buf))
        self.writer.write(length + buf)
        yield from self.writer.drain()
        status = []
        for i in range(len(jobs)):
            s = yield from self._recv_status()
            status.append(s)
        return status

    @asyncio.coroutine
    def measure(self,
                peer_ports,
                tx_mbps_l,
                traffics,
                randomize_payloads=None):
        if self.status != NFStatus.ready:
            log_error('Traffic generator is not ready')
            return
        if not self.reader or not self.writer:
            try:
                log_debug('Connected to MelvinGen')
                self.reader, self.writer = \
                    yield from asyncio.open_connection(self.host, self.port)
            except:
                self.reader = None
                self.writer = None
                log_error('Failed to connect to MelvinGen')
                return
        if not randomize_payloads:
            randomize_payloads = [False] * len(peer_ports)

        ports = [Port(p).neighbor_id if p else None for p in peer_ports]
        assert(len(ports) == len(tx_mbps_l) == len(traffics) ==
               len(randomize_payloads))

        duration_msec = config.duration_sec * 1000
        warmup_msec = config.warmup_sec * 1000

        jobs_traffic = []
        jobs_stats = [{"port": "00:00:00:00:00:00",
                       "stop": True,
                       "print": True}]

        # only last result will be reported
        for i, (port, tx_mbps, t, rnd) in \
                enumerate(zip(ports, tx_mbps_l, traffics, randomize_payloads)):
            if not port:
                continue
            traffic = Traffic(t)
            jobs_traffic.append({"port": port,
                                 "src_mac": "01:02:03:04:11:11",
                                 "dst_mac": "01:02:03:04:99:99",
                                 "tx_rate": int(tx_mbps),
                                 "duration": int(duration_msec),
                                 "warmup": int(warmup_msec),
                                 "num_flows": int(traffic.nr_flows),
                                 "size_min": int(traffic.size[0][0]),
                                 "size_max": int(traffic.size[0][0]),
                                 "life_min": duration_msec / 2,
                                 "life_max": duration_msec * 5,
                                 "randomize": rnd,
                                 "latency": True,
                                 "online": True})
        statuses = yield from self._send_jobs(jobs_traffic)
        for s, _ in statuses:
            if not s:
                return None
        yield from asyncio.sleep(config.duration_sec + 0.5)
        statuses = yield from self._send_jobs(jobs_stats)
        status, stats = statuses[0]
        if not status or not stats:
            return None

        return [stats[port] for port in ports]

    def stop(self):
        if self.writer:
            self.writer.close()
            self.reader = None
            self.writer = None
        super().stop()


class MoonGen(NFInstance, metaclass=Singleton):
    def __init__(self):
        super().__init__(name='moongen')
        raise Exception('broken')

    @asyncio.coroutine
    def measure(self, peer_ports, tx_mbps_l, traffics):
        if self.status != NFStatus.ready:
            log_error('Traffic generator is not ready')
            return

        ports = [Port(p).neighbor_id if p else None for p in peer_ports]

        # durationSec warmupSec portaddr rate nrFlows dist pktSize percent
        cmd = [str(config.duration_sec), str(config.warmup_sec)]
        for port, tx_mbps, t in zip(ports, tx_mbps_l, traffics):
            traffic = Traffic(t)
            cmd += [port, str(tx_mbps), str(traffic.nr_flows),
                    traffic.dist_spec, str(traffic.size[0][0]),
                    str(traffic.size[0][1])]
        cmd = ' '.join(cmd)
        log_debug(cmd)
        cmd += '\n'

        result = ''
        ignore = True
        while True:
            line = yield from self.proc.stdout.readline()
            line = line.decode().strip()

            if 'INPUT' in line:
                self.proc.stdin.write(cmd.encode())
            elif 'START' in line:
                ignore = False
                continue
            elif ignore or 'WARNING' in line:
                continue
            elif 'END' in line:
                ignore = True
                break
            else:
                result += line

        try:
            result = eval(result)
            return [result[i] for i in sorted(result.keys())]
        except:
            log_error(result)
            return None
