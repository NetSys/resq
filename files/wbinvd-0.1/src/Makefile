obj-m += wbinvd.o

all:
	make -C ${KERNELDIR} M=$(PWD)/src modules

install:
	make -C ${KERNELDIR} M=$(PWD)/src modules_install

clean:
	make -C ${KERNELDIR} M=$(PWD)/src clean

