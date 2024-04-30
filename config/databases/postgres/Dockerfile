FROM postgis/postgis:14-3.4
RUN sed -i -r 's/#huge_pages.*?/huge_pages = on/g' /usr/share/postgresql/postgresql.conf.sample