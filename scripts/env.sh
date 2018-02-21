#!/bin/sh

### ResQ Paths

export RESQ_PATH=/opt/resq
export RESQ_CFG_PATH=$RESQ_PATH/config
export RESQ_DATA_PATH=$RESQ_PATH/data
export RESQ_FILES_PATH=$RESQ_PATH/files
export RESQ_HUGE_PATH=/dev/hugepages
export RESQ_LOG_PATH=$RESQ_PATH/log
export RESQ_LXCROOTFS=/mnt/lxc.rootfs
export RESQ_NF_PATH=$RESQ_PATH/nf
export RESQ_PLOT_PATH=$RESQ_PATH/plot
export RESQ_SCRIPTS_PATH=$RESQ_PATH/scripts
export RESQ_TOOLS_PATH=/opt/tools

### Tools and Utilities

export GUROBI_HOME=$RESQ_TOOLS_PATH/gurobi701
export GRB_LICENSE_FILE=$GUROBI_HOME/gurobi.lic
export LINUX_SRC=$RESQ_TOOLS_PATH/linux
export MLC_BIN=$RESQ_TOOLS_PATH/intel/mlc/mlc_avx512
export PCM_MEMORY_BIN=$RESQ_TOOLS_PATH/intel/pcm/pcm-memory.x
export PCM_PCIE_BIN=$RESQ_TOOLS_PATH/intel/pcm/pcm-pcie.x
export PQOS_BIN=$RESQ_TOOLS_PATH/intel/intel-cmt-cat/pqos/pqos
export RDTSET_BIN=$RESQ_TOOLS_PATH/intel/intel-cmt-cat/rdtset/rdtset
export UCEVENT_BIN=$RESQ_TOOLS_PATH/pmu-tools/ucevent/ucevent.py

### DPDK

export RTE_VER=v17.05
export RTE_ARCH=x86_64
export RTE_SDK=$RESQ_NF_PATH/dpdk-$RTE_VER
export RTE_TARGET=resq

### NETMAP

export NETMAP_PATH=$RESQ_NF_PATH/netmap

### BESS

export BESS_PATH=$RESQ_NF_PATH/bess
export BESS_CONTROL_PORT=10514

### NFs

export BRO_PATH=$RESQ_NF_PATH/bro
export BRO_BIN=$BRO_PATH/src/bro

export CLICK_PATH=$RESQ_NF_PATH/click
export CLICK_BIN=$CLICK_PATH/$RTE_TARGET/userlevel/click
export CLICK_CFG_PATH=$RESQ_NF_PATH/config/click
export CLICK_CONTROL_PORT=20213

export SNORT_PATH=$RESQ_NF_PATH/snort
export SNORT_BIN=$SNORT_PATH/bin/snort
export SNORT_CFG_PATH=$RESQ_NF_PATH/config/snort

export SURICATA_PATH=$RESQ_NF_PATH/suricata
export SURICATA_BIN=$SURICATA_PATH/bin/suricata
export SURICATA_CFG_PATH=$RESQ_NF_PATH/config/suricata

### Traffic generators

export PKTGEN_PATH=$RESQ_NF_PATH/pktgen
export PKTGEN_BIN=$PKTGEN_PATH/app/app/$RTE_TARGET/pktgen

export MOONGEN_PATH=$RESQ_NF_PATH/moongen
export MOONGEN_BIN=$MOONGEN_PATH/build/MoonGen
export MOONGEN_LUA=$RESQ_SCRIPTS_PATH/moongen.lua

export BESSGEN_PATH=$RESQ_NF_PATH/trafficgen
export BESSGEN_BIN=$BESSGEN_PATH/run.py

export MELVINGEN_PATH=$RESQ_NF_PATH/melvingen
export MELVINGEN_BIN=$MELVINGEN_PATH/bin/pktgen
export MELVINGEN_HOST=c12
export MELVINGEN_CORES=4-23
#export MELVINGEN_HOST=c15
#export MELVINGEN_CORES=14-35
export MELVINGEN_PORT=5000

### Paths

export PYTHONPATH=$RESQ_PATH/python:$BESS_PATH
export LD_LIBRARY_PATH=$RESQ_NF_PATH/libhugetlbfs/obj64:$RESQ_NF_PATH/oisf/libhtp/htp/.libs:$GUROBI_HOME/lib:$RESQ_TOOLS_PATH/intel/intel-cmt-cat/lib:$LD_LIBRARY_PATH
export PATH=$PATH:$GUROBI_HOME/bin

