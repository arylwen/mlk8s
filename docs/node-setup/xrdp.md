# xrdp
#### <div style="text-align: left"> <a href="node-setup.md"><b>back to k8s cluster setup</b></a> <br/></div>

## install

```
sudo apt install xrdp 
sudo adduser xrdp ssl-cert 
```

## fix blank screen 
```
sudo nano /etc/xrdp/startwm.sh 
```
Add these lines just before the lines that test & execute Xsession as shown in the screenshot below. 

```
unset DBUS_SESSION_BUS_ADDRESS 
unset XDG_RUNTIME_DIR 
```

![fix blank screen](/docs/node-setup/images/xrdp.png)