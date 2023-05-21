# Prerequisites
  
    cd  mlk8s/config/postgres 

Create a file called password.txt and enter the password for the user pguser

    nano password.txt
 # microk8s   
Create the postgres deployment using mayastor on microk8s
    
    kubectl apply -f postgres-mayastor.yml 
    kubectl create secret --namespace=pgsql generic pgsql-password  --from-file=password=./password.txt 

## Build custom image
If you are using mayastor on microk8s, you enabled huge pages. Postgres will crash when huge pages are enabled, so they need to be disabled in the configuration. The code below builds the custom image and pushes it to the hub.docker.com registry:

    docker login 
    docker build --no-cache -t arylwen/postgis:14-3.3-huge-pages -f Dockerfile . 
    docker push arylwen/postgis:14-3.3-huge-pages 

# GKE 
Create the postgres deployment on google GKE
    
    kubectl create secret --namespace=default generic pgsql-password  --from-file=password=./password.txt 
    kubectl apply -f postgres.yml