# monitoring

The first step after we have a (hopefully) HA cluster set up is to have a way to get insights into its workings.

The easiest way is to enable the monitoring microk8s addon and set up OpenLens to take advantage of it.

It would allow a global view of the cluster. This would be the first place to check if you encouter issues: are all nodes on-line?
![global cluster view](/docs/monitoring/cluster%20status.png "Global cluster status.")

Or it would allow a global view of the deployments: are all applications up and running?
![global applicaiton view](/docs/monitoring/deployments.png)

To tun on monitoring run this on the first node you set-up for the cluster (rig1):

```
microk8s enable prometheus 
```

# cost monitoring

Follow the [steps here](/docs/monitoring/kubecost.md) to set up kubecost for the cluster.