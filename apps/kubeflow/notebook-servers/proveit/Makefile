TAG ?= $(shell git describe --tags --always --dirty)
REGISTRY ?= arylwen

docker-build-jupyter-pytorch-cuda-full:
	$(MAKE) docker-build-cuda -C ../jupyter-pytorch-full TAG=${TAG} 

docker-build: docker-build-jupyter-pytorch-cuda-full
	docker build -t ${REGISTRY}/proveit:${TAG} --build-arg BASE_IMG=${REGISTRY}/jupyter-pytorch-cuda-full:${TAG} -f Dockerfile .

docker-push:
	docker push ${REGISTRY}/proveit:${TAG}
