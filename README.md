# mlk8s
You can do this at home - personal ml cluster

## Why mlk8s
1. I wanted to make sense of my collection of articles: perhaps extract some triples, build a knowledge graph. I used ChatGPT and racked $10 between jesus h roosevelt christ and CTRL-C. 

2. I like to take classes. I think they provide a strucured approach to learn a topic. In one of the classes we had to build a semantic search engine. GKE was easy to get up to speed with, however the projected cost was $600/month, on the low end.
Can I build a mlk8s cluster to run a LLM, that I can afford?

3. Can I run k-means on the GPU?

4. Clustering surveys and representation learning: how does one represent categorical data in a continuous space so we can use k-means. Embedding space could be as big a 40,000 dimensions. How can we scale this horizontally? 

This repo contains step-by-step instructions to build a microk8s ml cluster from scratch. It features kubeflow, monitoring, storage and application examples.

## Step by step
1. [Node setup](/docs/node-setup/node-setup.md)
2. [Monitoring](/docs/monitoring/monitoring.md)
3. [Storage](/docs/storage/storage.md)
4. [Kubeflow](/docs/kubeflow/kubeflow.md)
5. [Databases](/docs/databases/databases.md)
5. [Devbox](/docs/devbox/devbox.md)
6. [Apps](/docs/apps/apps.md)

