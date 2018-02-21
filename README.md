# ResQ: Enabling SLOs in Network Function Virtualization
Network Function Virtualization is allowing carriers to replace dedicated middleboxes with Network Functions (NFs) consolidated on shared servers, but the question of how (and even whether) one can achieve performance SLOs with software packet processing remains open.
A key challenge is the high variability and unpredictability in throughput and latency introduced when NFs are consolidated.
We show that, using processor cache isolation and with careful sizing of I/O buffers, we can directly enforce a high degree of performance isolation among consolidated NFs -- for a wide range of NFs, our technique caps the maximum throughput degradation to 2.9% (compared to 44.3%), and the 95th percentile latency degradation to 2.5% (compared to 24.5%).
Building on this, we present ResQ, a resource manager for NFV that enforces performance SLOs for multi-tenant NFV clusters in a resource efficient manner.
ResQ achieves 60%-236% better resource efficiency for enforcing SLOs that contain contention-sensitive NFs compared to previous work.
The design of ResQ is discussed in an [NSDI paper](docs/resq_nsdi18.pdf).

This repository includes most of the code we developed for profiling, evaluation, and computing greedy and optimal schedules.
We hope, with minor modifications, it would also be useful as a framework for broader NFV experimentation.

## Hardware and OS
At least two machines are needed for experiments, once for hosting NFs and another for traffic generation.
The test machine must use a CPU that supports Intel Cache Allocation Technology, for example, an Intel Xeon E5 v4 series CPU (formerly known as Broadwell).
Ideally both machines should be dual socket with high core count processors.
We suggest to use the second processor of the NF machine for testing both for better isolation from the OS and higher port density:
The experiments often yield interesting results when large number of ports and cores are used.
NICs are to be directly attached to the CPU(s) that run the packet processing software.
Each port on the test machine is directly attached to the corresponding port on the traffic generator.
In our testbed, we use Intel Xeon E5 2695 v4 and E5 2658 v3 CPUs, and Intel XL710-QDA2 NICs.

Only Linux is supported and a kernel with RDT support is required (4.10+).
We have only tested our code on Debian 9 but it should be straightforward to use other GNU/Linux systems.

## Installation
Much of the codebase is in Python with helper bash scripts for setup and running NFs.
The bash script `scripts/setup.sh` contains routines for setting up dependencies.
To properly initialize variables that this setup script uses it is necessary to source `scripts/env.sh` beforehand.
You may customize `scripts/env.sh` if necessary.
It is necessary to set up passwordless SSH between the two machines.

NFs we used in our work rely on [DPDK](http://dpdk.org) or [netmap](https://github.com/luigirizzo/netmap) for I/O.
We use [BESS](https://github.com/netsys/bess) for port management -- not as a vSwitch.

We do not distribute the NFs we experimented with but you may add arbitrary NFs by adding proper configuration to the NFs manifest `config/nfs.toml`.
To allow for some flexibility for execution and monitoring of NFs, you may follow the pattern we use for most of our NFs and add helper routines to `scripts/nf_helper.sh`.

The pairwise connectivity between ports of the two servers must be captured in `config/ports.toml`: using LLDPd `scripts/discover_connectivity.sh` discovers connectivity and its output should be dumped to `config/ports.toml`.

## Running an experiment
First, ensure that you have sourced the environment setup script.
```
. scripts/env.sh
```

The main interface for experimentation is the `resq.run.Run` class which as input takes a list of NFs, a list of LLC allocation, a list of traffic profiles, and a list of utilizations.
The following snippet runs an experiment with two NFs each given 10% (2 out of 20 cache ways) of the LLC which are both fed input traffic consisting of packets uniformly sampled from 100k flows at random with no input rate limiting:
```
from resq.run import Run
from resq.util import ways_to_cbm

r = Run(['mazunat', 'mon'], cbms=[ways_to_cbm(2), ways_to_cbm(2, 2)], traffics=['u100k_60', 'u100k_60'], utilizations=[100, 100])
r.start()
```
More sample invocations could be found at `python/resq/profile/__init__.py`.
The results of experiments are persisted -- to force repeating the same experiment an argument of `force=True` may be passed to `start()`.

## References
* Amin Tootoonchian, Aurojit Panda, Chang Lan, Melvin Walls, Katerina Argyraki, Sylvia Ratnasamy, and Scott Shenker. ResQ: Enabling SLOs in Network Function Virtualization. In NSDI, 2018.
