#!/bin/bash

. ${0%/*}/env.sh

PEER=$1
[ -z "$PEER" ] && { echo Usage: $0 peer_host; exit 0; }

#lstopo -v | grep -B1 eth | sed -r 's/^\s+//g' | tr '\n' ' ' | sed -r -e 's/--//g' -e 's/PCI /\n&/g' -e 's/PCI[^=]+=//g' -e 's/ class=[^=]+=[^=]+=/ /g' | tr -d ')"' | awk '{print $3" "$2" "$1}' | sort -n | tail -n+2 | sed -e "s/^/\'/g" -e "s/$/\')/g" -e "s/ 68/': ('68/g" -e "s/ 00/', '00/g"

for prefix in "ssh $PEER" "eval"; do
    $prefix "$RESQ_SCRIPTS_PATH/bind_nics.sh kernel >/dev/null 2>&1"
    $prefix "find /sys/class/net -name 'e*' -execdir sh -c 'echo {} | sed s/^..//g | while read i; do ifconfig \$i up; done' ';'"
    $prefix "find /sys/kernel/debug/i40e -name command -exec sh -c 'echo lldp stop > {}' ';'"
    $prefix "service lldpd start"
done

sleep 2

for prefix in "ssh $PEER" "eval"; do
    $prefix "lldpcli update"
done

sleep 2

for bdf in $($RTE_SDK/usertools/dpdk-devbind.py --status | egrep 'XL710|X520-Q' | awk '{print $1}'); do
    [ -f /sys/kernel/debug/i40e/$bdf/command ] && echo lldp stop > /sys/kernel/debug/i40e/$bdf/command
    pcidir="/sys/bus/pci/devices/$bdf"
    driver=$(grep DRIVER $pcidir/uevent | sed s/DRIVER=//g)
    node=$(cat $pcidir/numa_node)
    hwaddr=$(cat $pcidir/net/*/address)
    iface=$(basename `ls -d $pcidir/net/*`)
    neighbor_chassis=$(lldpcli show neighbors ports $iface summary | grep SysName | awk '{print $NF}')
    neighbor_hwaddr=$(lldpcli show neighbors ports $iface summary | grep PortID | awk '{print $NF}' | tail -n1)
    neighbor_iface=$(lldpcli show neighbors ports $iface summary | grep PortDescr | awk '{print $NF}')
    [ -z $neighbor_iface ] && continue
    neighbor_pciaddr=$(ssh $PEER grep PCI_SLOT_NAME /sys/class/net/$neighbor_iface/device/uevent | awk -F'=' '{print $2}')
    neighbor_pcidir="/sys/bus/pci/devices/$neighbor_pciaddr"
    neighbor_node=$(ssh $PEER cat $neighbor_pcidir/numa_node)
    speed=$(cat /sys/class/net/$iface/speed)
    cat <<EOF
[$iface]
numa_node = $node
hwaddr = '$hwaddr'
pciaddr = '$bdf'
neighbor_chassis = '$neighbor_chassis'
neighbor_name = '$neighbor_iface'
neighbor_hwaddr = '$neighbor_hwaddr'
neighbor_pciaddr = '$neighbor_pciaddr'
neighbor_numa_node = '$neighbor_node'
dpdk_driver = 'igb_uio'
netmap_driver = '$driver'
speed = $speed
EOF
echo
done
