#!/bin/bash

. ${0%/*}/env.sh

usage() {
    echo "Usage: $0 <nf>" 1>&2
    exit 1
}

fail() {
    echo $1
    exit 1
}

ensure() {
    app=$1
    [ -z $app ] && usage

    local donefile=$RESQ_DATA_PATH/.done.$app
    [ -f $donefile ] && return
    $app && touch $donefile
}

bess() {
    ensure dpdk
    [ ! -d $BESS_PATH ] && git clone https://github.com/tootoonchian/bess.git $BESS_PATH
    cd $BESS_PATH
    ./build.py bess >/dev/null || fail "Failed to build BESS"
    return 0
}

click() {
    ensure dpdk
    git clone https://github.com/tootoonchian/click.git $CLICK_PATH
    cd $CLICK_PATH
    rsync -a $RESQ_FILES_PATH/click/ $CLICK_PATH/
    mkdir $RTE_TARGET
    cd $RTE_TARGET
    ../configure CXXFLAGS="-std=c++11 -O3 -g -DNDEBUG" --disable-linuxmodule --enable-ipsec --enable-user-multithread --with-dpdk >/dev/null || fail "Failed to configure Click"
    make -j >/dev/null || fail "Failed to build Click"
    return 0
}

dpdk() {
    git clone http://dpdk.org/git/dpdk $RTE_SDK
    cd $RTE_SDK
    git reset --hard $RTE_VER
    rsync -a $RESQ_FILES_PATH/dpdk-$RTE_VER/ $RTE_SDK/
    cat >$RTE_SDK/config/defconfig_$RTE_TARGET <<EOF
#include "common_linuxapp"

CONFIG_RTE_MACHINE="native"

CONFIG_RTE_ARCH="x86_64"
CONFIG_RTE_ARCH_X86_64=y
CONFIG_RTE_ARCH_X86=y
CONFIG_RTE_ARCH_64=y

CONFIG_RTE_TOOLCHAIN="gcc"
CONFIG_RTE_TOOLCHAIN_GCC=y

CONFIG_RTE_BUILD_COMBINE_LIBS=y

CONFIG_RTE_APP_TEST=n
CONFIG_RTE_KNI_KMOD=n

CONFIG_RTE_LIBRTE_CXGBE_PMD=n
CONFIG_RTE_LIBRTE_EM_PMD=n
CONFIG_RTE_LIBRTE_ENIC_PMD=n
CONFIG_RTE_LIBRTE_IEEE1588=n
CONFIG_RTE_LIBRTE_IGB_PMD=n
CONFIG_RTE_LIBRTE_KNI=n
CONFIG_RTE_LIBRTE_VIRTIO_PMD=n
CONFIG_RTE_LIBRTE_VMXNET3_PMD=n
EOF
    make -j install T=$RTE_TARGET >/dev/null || fail "Failed to compile DPDK"
    return 0
}

dep() {
    apt install python3 python3-pip

    pip3 install --upgrade \
        colorlog cycler cython flake8 humanfriendly ipaddress kmod \
        matplotlib msgpack-python netifaces networkx numpy pandas \
        pip pcapy protobuf psutil pyudev scipy seaborn simplejson \
        sqlalchemy sqlalchemy_utils toml virtfs
    return 0
}

linux() {
    git clone https://github.com/fyu1/linux.git $LINUX_SRC
    cd $LINUX_SRC
    git checkout cat16.1
    cp /boot/config-`uname -r` .config
    make olddefconfig
    make-kpkg --append-to-version "-cat" --revision $(date +"%Y%m%d") kernel_image kernel_headers
    echo "Please install the generated kernel image and header packages"
    read
    return 0
}

lxc() {
    cat >$RESQ_CFG_PATH/resq/lxc.conf <<EOF
lxc.init_cmd = /usr/sbin/init.lxc -- sleep infinity
lxc.console = none
lxc.rootfs = $RESQ_LXCROOTFS

lxc.aa_profile = unconfined
lxc.autodev = 0
#lxc.hook.autodev = $RESQ_SCRIPTS_PATH/mknod_hook.sh
lxc.cgroup.devices.allow = a
lxc.cgroup.devices.allow = c *:* rwm
lxc.cgroup.devices.allow = b *:* rwm

lxc.mount.auto = cgroup sys:rw
lxc.mount.entry = /$RESQ_PATH $RESQ_PATH none defaults,bind,create=dir 0 0
lxc.mount.entry = /$HUGE_PATH $HUGE_PATH none defaults,bind,create=dir 0 0
lxc.mount.entry = proc proc proc nosuid,nodev,noexec,create=dir  0 0
lxc.mount.entry = shm dev/shm tmpfs rw,nosuid,nodev,noexec,relatime,mode=1777,size=256m,create=dir 0 0
#lxc.mount.entry = tmpfs run tmpfs nosuid,nodev,noexec,mode=0755,size=128m,create=dir 0 0
lxc.mount.entry = tmpfs tmp tmpfs nosuid,nodev,noexec,mode=1777,size=128m,create=dir 0 0
lxc.mount.entry = /dev dev none defaults,bind,create=dir 0 0
lxc.mount.entry = /bin bin none defaults,bind,create=dir 0 0
lxc.mount.entry = /etc etc none defaults,bind,create=dir 0 0
lxc.mount.entry = /lib lib none defaults,bind,create=dir 0 0
lxc.mount.entry = /lib64 lib64 none defaults,bind,create=dir 0 0
lxc.mount.entry = /run run none defaults,bind,create=dir 0 0
lxc.mount.entry = /sbin sbin none defaults,bind,create=dir 0 0
lxc.mount.entry = /usr usr none defaults,bind,create=dir 0 0
lxc.mount.entry = /var var none defaults,bind,create=dir 0 0
EOF
    return 0
}

melvingen() {
    ensure dpdk
    [ ! -d $MELVINGEN_PATH ] && git clone https://github.com/tootoonchian/pktgen.git $MELVINGEN_PATH
    cd $MELVINGEN_PATH
    make -j >/dev/null || fail "Failed to compile MelvinGen"
    return 0
}

netmap() {
    [ ! -d $NETMAP_PATH ] && git clone https://github.com/luigirizzo/netmap.git $NETMAP_PATH
    cd $NETMAP_PATH/LINUX
    ./configure --driver-suffix=-netmap --drivers=ixgbe,i40e --kernel-sources=$LINUX_SRC || fail "Failed to configure netmap"
    make -j install || fail "Failed to compile netmap"
    return 0
}

snort() {
    local DAQ_PATH=$SNORT_PATH/daq
    mkdir -p $SNORT_PATH $DAQ_PATH
    wget https://www.snort.org/downloads/snort/snort-2.9.8.3.tar.gz -P /tmp
    wget https://www.snort.org/downloads/snort/daq-2.0.6.tar.gz -P /tmp
    tar xf /tmp/snort-2.9.8.3.tar.gz -C $SNORT_PATH --strip-components 1
    tar xf /tmp/daq-2.0.6.tar.gz -C $DAQ_PATH --strip-components 1
    rsync -a $RESQ_FILES_PATH/snort/ $SNORT_PATH/
    cd $DAQ_PATH
    CFLAGS="-O2" CXXFLAGS="-O2" ./configure --disable-nfq-module --disable-ipq-module --with-netmap-includes=$NETMAP_PATH/sys --prefix=$DAQ_PATH --exec-prefix=$DAQ_PATH >/dev/null || fail "Failed to configure DAQ"
    make -j install >/dev/null
    make -j install >/dev/null || fail "Failed to compile DAQ"
    cd $SNORT_PATH
    CFLAGS="-O2" CXXFLAGS="-O2" ./configure --enable-sourcefire --prefix=$SNORT_PATH --exec-prefix=$SNORT_PATH --sysconfdir=$SNORT_CFG_PATH/.. --with-daq-includes=$DAQ_PATH/include --with-daq-libraries=$DAQ_PATH/lib >/dev/null || fail "Failed to configure Snort"
    make -j install >/dev/null || fail "Failed to compile Snort"
    sed -e "s?%SNORT_CFG_PATH%?$SNORT_CFG_PATH?g" \
        -e "s?%SNORT_PATH%?$SNORT_PATH?g" \
        -e "s?%RESQ_LOG_PATH%?$RESQ_LOG_PATH?g" \
        $SNORT_CFG_PATH/snort.conf.in > $SNORT_CFG_PATH/snort.conf
    return 0
}

suricata() {
    apt install libnet1-dev libgeoip-dev libmagic-dev libyaml-dev libhtp-dev libdumbnet-dev
    git clone https://github.com/inliniac/suricata.git $SURICATA_PATH
    cd $SURICATA_PATH
    git reset --hard suricata-3.1
    git clone https://github.com/OISF/libhtp
    ./autogen.sh >/dev/null || fail "Failed to run Suricata autogen.sh"
    ./configure --enable-netmap --with-netmap-includes=$NETMAP_PATH/sys --prefix=$SURICATA_PATH --exec-prefix=$SURICATA_PATH --sysconfdir=$SURICATA_CFG_PATH/.. >/dev/null || fail "Failed to configure Suricata"
    make -j install >/dev/null || fail "Failed to compile Suricata"
    sed -e "s?%SURICATA_CFG_PATH%?$SURICATA_CFG_PATH?g" \
        -e "s?%SURICATA_PATH%?$SURICATA_PATH?g" \
        -e "s?%RESQ_LOG_PATH%?$RESQ_LOG_PATH?g" \
        $SURICATA_CFG_PATH/suricata.yaml.in > $SURICATA_CFG_PATH/suricata.yaml
    return 0
}

tools() {
    mkdir $RESQ_TOOLS_PATH >/dev/null 2>&1
    cd $RESQ_TOOLS_PATH
    echo "Please manually download the latest version of Intel MLC from https://software.intel.com/en-us/articles/intelr-memory-latency-checker and place it at $RESQ_TOOLS_PATH/mlc"
    read
    echo "Please manually download the latest version of Intel PCM from https://software.intel.com/en-us/articles/intel-performance-counter-monitor and place it at $RESQ_TOOLS_PATH/pcm"
    read
    git clone https://software.intel.com/en-us/articles/intelr-memory-latency-checker $RESQ_TOOLS_PATH/mlc
    cd $RESQ_TOOLS_PATH/intel-cmt-cat
    git clone https://github.com/andikleen/pmu-tools/ $RESQ_TOOLS_PATH/pmu-tools
    return 0
}

wbinvd() {
    rsync -a $RESQ_FILES_PATH/wbinvd-0.1 /usr/src
    dkms add -m wbinvd -v 0.1 || fail "Failed to add wbinvd to DKMS"
    dkms build -m wbinvd -v 0.1 || fail "Failed to build wbinvd through DKMS"
    dkms install -m wbinvd -v 0.1 || fail "Failed to install wbinvd through DKMS"
    return 0
}

app=$1

[ "$app" = "ensure" ] && fail "ensure is an invalid name"

ensure $app
