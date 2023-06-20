DOCKER_FILE=Dockerfile
DOCKER_TAG=v1.0.0
DOCKER_IMAGE=flyte-nlp
DOCKER_FQN=localhost:30000/${DOCKER_IMAGE}:${DOCKER_TAG}

.PHONY: setup
setup:
	flytectl demo start

.PHONY: docker
docker:
	docker build --no-cache -f ${DOCKER_FILE} -t ${DOCKER_FQN} .
	docker push ${DOCKER_FQN}

.PHONY: register
register:
	poetry run pyflyte register --project nlpprocessing --domain development --image ${DOCKER_FQN} flyte_nlp/
