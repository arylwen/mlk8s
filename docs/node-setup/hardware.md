# Hardware setup
#### <div style="text-align: left"> <a href="node-setup.md"><b>back to k8s cluster setup</b></a> <br/></div>

Install Ubuntu 20.4 on each node.

## networking
k8s is very sensitive to the network quality. 10ms ping time is ok; 200ms ping time is not. mikrok8s works with dqlite and microk8s 1.24 has a bug where cpu utilization spikes to 100% in a multinode scenario when dqlite on different nodes becomes out of sync.

Ideally all nodes would be on the same switch, connected through ethernet cables. Use wifi only when absolutely needed.

Attempt to have at least the 3 master nodes in a HA setup connected through an ethernet cable to a switch.

### hack
If you cannot have an ethernet cable at the cluster location, you could use a wifi network extender with an ethernet outlet.

