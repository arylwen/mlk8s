TAG ?= $(shell git describe --tags --always --dirty)
REGISTRY ?= arylwen

docker-build-base:
	$(MAKE) docker-build -C ../base TAG=${TAG}

docker-build: docker-build-base
	docker build -t ${REGISTRY}/jupyter:${TAG} --build-arg BASE_IMG=${REGISTRY}/base:${TAG} -f Dockerfile . 

docker-push:
	docker push ${REGISTRY}/jupyter:${TAG}