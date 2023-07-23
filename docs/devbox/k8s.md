# k8s

## install kubectl

```
sudo mkdir /etc/apt/keyrings
sudo apt-get update
sudo apt-get install -y ca-certificates curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-archive-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl

```

You can find more details [here](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-using-native-package-management).

## kubeconfig

kubeconfig allows the connection to your k8s cluster

```
cd ~
mkdir .kube
```
Copy cluster's config file in ~/.kube. Verify cluster configuation:

```
kubectl config view 
```
        
<pre>
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: DATA+OMITTED
    server: https://10.0.0.xx:16443
  name: microk8s-cluster
contexts:
- context:
    cluster: microk8s-cluster
    user: admin
  name: microk8s
current-context: microk8s
kind: Config
preferences: {}
users:
- name: admin
  user:
    token: REDACTED
</pre>

## open lens

### download
https://github.com/MuhammedKalkan/OpenLens/releases

## move to applications 
```
cd  ~/Downloads
mv OpenLens-6.5.2-366.x86_64.AppImage ~/.local/share/applications/
```

## create desktop shortcut
```
cd ~/.local/share/applications/
nano OpenLens.desktop
```

Paste the contents below and save. You should be able to find OpenLens in "Show Applications" and add it to favorites/dock.
```
[Desktop Entry]
Name=OpenLens
Comment=Working with k8s has never been so easy and convenient.
Exec="/home/mlk8s/.local/share/applications/OpenLens-6.5.2-366.x86_64.AppImage" %U
Terminal=false
Type=Application
Icon=/home/mlk8s/.local/share/applications/lens-logo-icon.svg
Categories=Development;
TryExec=/home/mlk8s/.local/share/applications/OpenLens-6.5.2-366.x86_64.AppImage
```

## node and pod menu

Navigate to hamburger menu->extensions. In the extensions box enter:
```
@alebcay/openlens-node-pod-menu
```

https://github.com/alebcay/openlens-node-pod-menu

## lens resource map

Navigate to hamburger menu->extensions. In the extensions box enter:
```
@nevalla/kube-resource-map
```

https://github.com/nevalla/lens-resource-map-extension