# /etc/hosts

#### <div style="text-align: left"> <a href="node-setup.md"><b>back to k8s cluster setup</b></a> <br/></div>

Ideally /etc/hosts should be the same on all server nodes. It would help to have the servers on the devbox also. Probably ok to do this at home.

```
127.0.0.1 localhost 
::1             localhost 
#dev box
10.0.0.100      devbox 
# HA master nodes
10.0.0.101      rig1 
10.0.0.102      rig2 
10.0.0.103      rig3 
# GPU worker node
10.0.0.110      rig4 
# storage, local registry and other services - GPU box gets relly busy
10.0.0.120      rig5 

#local hostnames 
10.0.0.102      milvus-attu.local 
10.0.0.102      prometheus-ui.local 
10.0.0.102      grafana-ui.local 
10.0.0.102      redis-commander.local  
10.0.0.102      semsearch-ui.local 
10.0.0.102      k8s-dashboard.local 

#services 
10.0.0.102      minio 
10.0.0.102      milvus   
10.0.0.102      redis   
10.0.0.102      pgsql14   
10.0.0.102      semsearch-api 
10.0.0.102      registry.local 

# The following lines are desirable for IPv6 capable hosts 
fe00::0 ip6-localnet 
ff00::0 ip6-mcastprefix 
ff02::1 ip6-allnodes 
ff02::2 ip6-allrouters 

```