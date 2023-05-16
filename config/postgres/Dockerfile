FROM postgis/postgis:14-3.3
RUN sed -i -r 's/#huge_pages.*?/huge_pages = off/g' /usr/share/postgresql/postgresql.conf.sample