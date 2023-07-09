# mlk8s - personal ml cluster
You could do this at home, [get started here](/docs/devbox/devbox.md).
## why mlk8s
1. I wanted to make sense of my collection of articles: perhaps extract some triples, build a knowledge graph. I used llama index with OpenAI ChatGPT and racked $10 between jesus h roosevelt christ and CTRL-C. 

2. I like to take classes. I think they provide a strucured approach to learn a topic. In one of the classes we had to build a semantic search engine. GKE was easy to get up to speed with, however the projected cost was $600/month, on the low end.
Can I build a mlk8s cluster to run a LLM, that I can afford?

3. Can I run k-means on the GPU?

4. Clustering surveys and representation learning: how does one represent categorical data in a continuous space so we can use k-means. Embedding space could be as big a 40,000 dimensions. How can we scale this horizontally? 

This repo contains step-by-step instructions to build a microk8s ml cluster from scratch. It features kubeflow, monitoring, storage and application examples.

## step by step
1. [node setup](/docs/node-setup/node-setup.md)
2. [monitoring](/docs/monitoring/monitoring.md)
3. [storage](/docs/storage/storage.md)
4. [kubeflow](/docs/kubeflow/kubeflow.md)
5. [databases](/docs/databases/databases.md)
5. [devbox](/docs/devbox/devbox.md)
6. [apps](/docs/apps/apps.md)

