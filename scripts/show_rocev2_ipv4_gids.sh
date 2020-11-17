#!/bin/bash
#Auther: marlonfeng@tencent.com 2020/11/10

TARGET_TYPE="v2"

DEVS=$1
if [ -z "$DEVS" ] ; then
	DEVS=$(ls /sys/class/infiniband/)
fi
#echo -e "===RoCE V2 IPv4 GIDS==="
#echo -e "DEV\t\tGID"
#echo -e "------\t\t---\t"
for d in $DEVS ; do
	for p in $(ls /sys/class/infiniband/$d/ports/) ; do
		for g in $(ls /sys/class/infiniband/$d/ports/$p/gids/) ; do
			gid=$(cat /sys/class/infiniband/$d/ports/$p/gids/$g);
			if [ $gid = 0000:0000:0000:0000:0000:0000:0000:0000 ] ; then
				continue
			fi
			if [ $gid = fe80:0000:0000:0000:0000:0000:0000:0000 ] ; then
				continue
			fi
			_ndev=$(cat /sys/class/infiniband/$d/ports/$p/gid_attrs/ndevs/$g 2>/dev/null)
			__type=$(cat /sys/class/infiniband/$d/ports/$p/gid_attrs/types/$g 2>/dev/null)
			_type=$(echo $__type| grep -o "[Vv].*")
			if [ $(echo $gid | cut -d ":" -f -1) = "0000" ] ; then
				if [ $_type == $TARGET_TYPE ]; then
				    echo -e "$g"
				fi
			fi
		done #g (gid)
	done #p (port)
done #d (dev)