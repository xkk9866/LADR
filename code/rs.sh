#!/bin/bash
# retrying ssh: rs.sh "<remote command>"
for i in 1 2 3 4 5 6; do timeout ${T:-600} ssh -o ConnectTimeout=20 srv "$1" && exit 0; rc=$?; [ $rc -ne 255 ] && exit $rc; sleep 8; done; exit 255
