#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/smp.h>
#include <linux/version.h>

static int __init wbinvd_init(void) {
    printk(KERN_INFO "invalidating caches on all cpus");
    wbinvd_on_all_cpus();
    return 0;
}

static void __exit wbinvd_exit(void) {}

module_init(wbinvd_init);
module_exit(wbinvd_exit);
