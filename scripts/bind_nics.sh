#!/bin/bash
MODE=${1:-kernel}
DPDK_DRIVER=${2:-vfio-pci}
[ $MODE != kernel -a $MODE != dpdk ] && echo "Usage: $0 [kernel|dpdk]" 1>&2 && exit 1
if [ $MODE = kernel ]; then
    $RTE_SDK/usertools/dpdk-devbind.py -b i40e $($RTE_SDK/usertools/dpdk-devbind.py --status | egrep 'XL710' | awk '{print $1}')
    $RTE_SDK/usertools/dpdk-devbind.py -b ixgbe $($RTE_SDK/usertools/dpdk-devbind.py --status | egrep 'X520-Q' | awk '{print $1}')
    $RTE_SDK/usertools/dpdk-devbind.py -b fm10k $($RTE_SDK/usertools/dpdk-devbind.py --status | egrep 'FM10420' | awk '{print $1}')
elif [ $MODE = dpdk ]; then
    modprobe $DPDK_DRIVER
    $RTE_SDK/usertools/dpdk-devbind.py -b $DPDK_DRIVER $($RTE_SDK/usertools/dpdk-devbind.py --status | egrep '(X520-Q|XL710|FM10420)' | awk '{print $1}')
fi
