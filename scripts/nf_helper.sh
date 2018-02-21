#!/bin/bash

. ${0%/*}/env.sh

usage() {
    echo "Usage: $0 <subcmd> -n <nf_name> [-h <handle>] [-i <portname/pciaddr>] [-c <core>] [-r <remotehost] [-b <burst>] [-f]" 1>&2
    exit 1
}

getpids() {
    local pid=$(cat $pidfile)
    pgrep -wP $pid -d' '
}

start() {
    set -m
    local log_str

    stop &>/dev/null
    start_cmd &>/dev/null
    echo "$start_cmd_str" > $logfile
    [ $foreground -eq 1 ] && local fg_arg="-f"

    eval daemon -U --name "$handle" --pidfile "$pidfile" -o "$logfile" $fg_arg -- $start_cmd_str
    #eval "$start_cmd_str >> $logfile 2>&1 &"
    #echo $! > $pidfile

    #if [ $foreground -eq 1 ]; then
    #    tail -f $logfile & fg %1
    #fi
}

start_cmd() {
    local iportname=${portnames[0]}
    local iportaddr=${portaddrs[0]}
    local oportname=${portnames[1]:-$iportname}
    local oportaddr=${portaddrs[1]:-$iportaddr}
    local socket=-1
    [ -z "$nf" ] && usage
    [ ! -z "$core" ] && socket=$(cat /sys/devices/system/cpu/cpu$core/topology/physical_package_id)
    local cmd
    #cmd="HUGETLB_SHARE=1 HUGETLB_MORECORE=yes LD_PRELOAD=libhugetlbfs.so"
    cmd="$cmd stdbuf -o 0 -e 0"
    #[ $nf != "bess" ] && cmd="$cmd numactl -m $socket -N $socket -C $core"

    # [ ${#iportnames[@]} -gt 0 ]
    case "$nf" in
        mlc*)
            local idle_cycles=${nf:3}
            local cpumask=$(printf "%x" `echo 2^$core | bc`)
            cmd="$MLC_BIN $args --loaded_latency -T -t9999 -k$core -d$idle_cycles -W10 -Y"
            ;;
        suricata)
            cmd="$SURICATA_BIN $args -c $SURICATA_CFG_PATH/suricata.yaml --netmap=$iportname --set netmap.0.copy-iface=$oportname --init-errors-fatal --runmode single -k none"
            ;;
        snort)
            socket_mem="100"
            [ "$socket" = 0 ] && socket_mem="100,0"
            [ "$socket" = 1 ] && socket_mem="0,100"

            cmd="$SNORT_BIN $args -c $SNORT_CFG_PATH/snort.conf -A fast -y --daq-dir /usr/local/lib/daq --daq dpdk --daq-var dpdk_args='--proc-type secondary --file-prefix rte -n 4 --socket-mem $socket_mem -l $core' -i $iportaddr,$oportaddr --daq-mode inline -Q -k none"
            #cmd="$SNORT_BIN $args -c $SNORT_CFG_PATH/snort.conf -A fast -y -i $iportname:$oportname --daq netmap --daq-mode inline -Q -k none"
            ;;
        bro)
            cmd="BROPATH=`./bro-path-dev` $args $BRO_BIN -i $iportname local 'Site::local_nets += { 169.0.0.0/8 }'"
            ;;
        bess)
            cmd="$BESS_PATH/bin/bessd $args -f -k"
            ;;
        bessgen)
            cmd="$BESSGEN_BIN"
            ;;
        melvingen)
            cmd="$MELVINGEN_BIN $args -- $MELVINGEN_PORT"
            ;;
        moongen)
            cmd="$MOONGEN_BIN $args $MOONGEN_LUA"
            ;;
        *)
            socket_mem="100"
            [ "$socket" = 0 ] && socket_mem="100,0"
            [ "$socket" = 1 ] && socket_mem="0,100"

            cmd="$CLICK_BIN $args -C $CLICK_CFG_PATH -p $CLICK_CONTROL_PORT -e 'require(library nsdi12.click); DPDKInfo(CORES $core, PREFIX rte, HUGE_DIR $RESQ_HUGE_PATH, NR_CHANNELS 4, SOCKET_MEM \"$socket_mem\", DEVS \""

            if [ "$nf" = 'dpdk_primary' ]; then
                cmd="$cmd\");'"
            else
                cmd="$cmd $iportname($iportaddr)"
                [ $iportname != $oportname ] && cmd="$cmd $oportname($oportaddr)"
                cmd="$cmd\");"
                cmd="$cmd in :: FromDevice($iportname, BURST $burst, METHOD DPDK, PROMISC true);"
                cmd="$cmd out :: ToDevice($oportname, BURST $burst, METHOD DPDK, YIELD_USEC 0);"
                cmd="$cmd in -> $nf -> SimpleQueue($((burst * 2))) -> out;'"
            fi
            ;;
    esac

    echo $cmd
    start_cmd_str=$cmd
}

status() {
    ready=0
    running=0
    daemon --running --name "$handle" --pidfile "$pidfile" && running=1
    [ $running -eq 0 ] && { echo STOPPED; return; }

    local pid=$(cat $pidfile)
    local nthreads=$(pgrep -wP $pid -c)

    case "$nf" in
        mlc*)
            [ $nthreads -ge 3 ] && ready=1
            ;;
        snort)
            [ $nthreads -ge 2 ] && ready=1
            ;;
        suricata)
            [ $nthreads -ge 6 ] && ready=1
            ;;
        melvingen)
            netstat -nap | grep -q :${MELVINGEN_PORT}.*LISTEN && ready=1
            ;;
        bess)
            netstat -nap | grep -q :${BESS_CONTROL_PORT}.*LISTEN && ready=1
            ;;
        *)
            netstat -nap | grep -q :${CLICK_CONTROL_PORT}.*LISTEN && ready=1
            ;;
    esac
    if [ $ready -eq 1 ]; then
        # extra check to ensure links are up
        for p in ${portnames[@]}; do
            local d = "/sys/class/net/$p"
            [ -d $d -a `cat $d/operstate` != 'up' ] && ready=0 && break
        done
    fi
    [ $ready -eq 1 ] && { echo READY; return; }
    [ $running -eq 1 ] && { echo RUNNING; return; }
}

stop() {
    daemon --stop --name "$handle" --pidfile "$pidfile"
    #local pid=$(cat $pidfile)
    #[ ! -z "$pid" ] && kill -9 $pid &>/dev/null
}

cmd=$1
shift

burst=32
foreground=0
nr_ports=0
portnames=
portaddrs=

while getopts "h:n:i:c:b:r:f" o; do
    case "${o}" in
        h)
            handle=${OPTARG}
            ;;
        n)
            nf=${OPTARG}
            ;;
        i)
            portnames[nr_ports]=${OPTARG%/*}
            portaddrs[nr_ports]=${OPTARG#*/}
            nr_ports=$((nr_ports + 1))
            ;;
        c)
            core=${OPTARG}
            ;;
        b)
            burst=${OPTARG}
            ;;
        r)
            remotehost=${OPTARG}
            ;;
        f)
            foreground=1
            ;;
        *)
            usage
            ;;
    esac
done

[ -z "$cmd" -o -z "$nf" ] && usage

if [ ! -z "$remotehost" -a "$remotehost" != "$HOSTNAME" ]; then
    ssh -q -t $remotehost -- $0 $cmd $@
    exit $?
fi

if [ -z "$handle" ]; then
    if [ -z ${ports[0]} ]; then
        handle="nf-${nf}"
    else
        handle="nf-${ports[0]}"
    fi
fi

logfile=$RESQ_LOG_PATH/$handle.log
pidfile=/var/run/$handle.pid

shift $((OPTIND-1))
args=$@

$cmd
