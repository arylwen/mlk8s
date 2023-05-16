
    cd  mlk8s/config/postgres 

Create a file calles password.txt and enter the password for the user pguser

    nano password.txt
 
Create the postgres deployment on google GKE
    
    kubectl create secret --namespace=default generic pgsql-password  --from-file=password=./password.txt 
    kubectl apply -f postgres.yml
    
Create the postgres deployment using mayastor
    
    kubectl apply -f postgres-mayastor.yml 
    kubectl create secret --namespace=pgsql generic pgsql-password  --from-file=password=./password.txt 

If you are using mayastor, you enabled huge pages. Postgres will crash when huge pages are enables, so they need to be disabled in the configuration. If you want to build the custom image and pus it to a different registry:

    docker login 
    docker build --no-cache -t arylwen/postgis:14-3.3-huge-pages -f Dockerfile . 
    docker push arylwen/postgis:14-3.3-huge-pages 