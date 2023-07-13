# microk8s node prep

microk8s starts with one node. You can probably run the whole cluster on one machine, given enough hardware for your application. Ideally, set up a high availability (HA) cluster with 3 master nodes. Configure the ML nodes as worker nodes. ML processing is resource intensive and consumer hardware tends to crash under load. The HA mode would preserve the state of the cluster and help with its stability.

## data folder

Create a root owned /data/microk8s folder on the node for openebs. All openebs volumes will be stored here. 

```
sudo mkdir /data 
sudo mkdir /data/microk8s 
```

## hosts
Update /etc/hosts on each node. Add and entry to hosts for each node. 

```
10.0.0.xx      rig1
10.0.0.xx      rig2 
```

## DNS
 
### resolvconf 
```
sudo apt install -y resolvconf 
sudo systemctl enable --now resolvconf.service 
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolvconf/resolv.conf.d/head 
sudo resolvconf -u 
```

## storage

### OpenEBS pre-requisite 
```
sudo apt-get update 
sudo apt-get install open-iscsi 
sudo systemctl enable --now iscsid 
systemctl status iscsid  
```

### mayastor prerequisite 
```
sudo sysctl vm.nr_hugepages=1024 
echo 'vm.nr_hugepages=1024' | sudo tee -a /etc/sysctl.conf 
sudo apt install -y linux-modules-extra-$(uname -r) 
sudo modprobe nvme_tcp 
echo 'nvme-tcp' | sudo tee -a /etc/modules-load.d/microk8s-mayastor.conf 
sync 
sudo reboot 
```

# build the cluster
A microk8s node is a cluster. 2 microk8s nodes are a cluster. One extends a microk8s cluster by adding nodes
 
## create the first node - rig1
The first node in a microk8s cluster is a master node. We create a microk8s cluster by simply installing microk8s. Kubeflow requires k8s 1.24.
Let's call the first master node **rig1**.

```
sudo snap install microk8s --classic --channel=1.24/stable 
```

#### enable microk8s command 

```
sudo usermod -a -G microk8s arylwen 
sudo chown -f -R arylwen ~/.kube 
newgrp microk8s 
```

## add another master node - rig2
On the target node, e.g. **rig2**, install microk8s  
```
sudo snap install microk8s --classic --channel=1.24/stable 
```
### enable microk8s command 
```
sudo usermod -a -G microk8s arylwen 
sudo chown -f -R arylwen ~/.kube 
newgrp microk8s 
```
 
## join rig2 to the cluster
### on rig1 – after installing microk8s on target node (rig2) because of token expiration 
```
microk8s add-node 
```
 
### on target node, e.g. rig2 

#the token expires in about 5 mins so needs a freshly generated token 
```
microk8s join 10.0.0.xx:25000/bf4ef68d5bcfc9c6ff2d1ff5fddad317/31e9f275071d 
```

## on rig1
```
microk8s kubectl get nodes 
```
 
# patch for calico issue 
```
kubectl patch ds calico-node -n kube-system -p '{"spec":{"template":{"spec":{"containers":[{"name":"calico-node","image":"docker.io/calico/node:v3.22.3"}]}}}}' 
```
https://github.com/dyrnq/kubeadm-vagrant/issues/33  

# remove node
If a node becomes unavailable, e.g. crashes, it can be removed with:
```
microk8s remove-node rig2 –force 
```