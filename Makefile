DOCKER_FILE=Dockerfile
DOCKER_TAG=v1.0.0
DOCKER_IMAGE=flyte-nlp
DOCKER_FQN=localhost:30000/${DOCKER_IMAGE}:${DOCKER_TAG}

.PHONY: docker

docker:
	docker build -f ${DOCKER_FILE} -t ${DOCKER_FQN} .
	docker push ${DOCKER_FQN}
