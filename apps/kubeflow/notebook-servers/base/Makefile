TAG ?= $(shell git describe --tags --always --dirty)
REGISTRY ?= arylwen

docker-build:
	docker build -t ${REGISTRY}/base:${TAG} -f Dockerfile .

docker-push:
	docker push ${REGISTRY}/base:${TAG}
