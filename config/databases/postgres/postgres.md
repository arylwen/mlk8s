# Prerequisites
  
    cd  mlk8s/config/databases/postgres 

Create a file called password.txt and enter the password for the postgresql user pguser.

## Build custom image
Huge pages could enable better postgresql performance. The code below builds the custom image and pushes it to the hub.docker.com registry:

    docker login 
    docker build --no-cache -t arylwen/postgis:14-3.4-huge-pages -f Dockerfile . 
    docker push arylwen/postgis:14-3.4-huge-pages 

## Create namespace
    kubectl apply -f namespace.yml 

## Set pguser password
    nano password.txt
    kubectl create secret --namespace=pgsql generic pgsql-password  --from-file=password=./password.txt 

 # microk8s   
Create the postgres deployment using the openebs-hostpath storage class on microk8s.

    kubectl apply -f postgres-openebs-hostpath.yml 

