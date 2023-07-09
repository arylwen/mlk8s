# docker
Docker is needed to build container images. Snap install ismore secure, can only run in $HOME. 

```
sudo snap install docker 
```

```
sudo groupadd docker 
sudo usermod -aG docker $USER 
newgrp docker 
sudo chown root:docker /var/run/docker.sock 
```
 
Verify:
```
docker image list 
```