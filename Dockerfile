FROM python:3.9-slim-buster

WORKDIR /root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get update && apt-get install -y build-essential

# Virtual environment
# RUN python3 -m venv ${VENV}
# ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
# COPY requirements.txt /root
# RUN pip install -r /root/requirements.txt

# Copy the actual code
COPY . /root

# Install Poetry
# RUN curl -sSL https://install.python-poetry.org | python3 -
# ENV PATH="$HOME/.local/bin:$PATH"
RUN pip3 install poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev

# Test the code
RUN poetry run python -c "from flyte_nlp.workflows.word2vec_and_lda import nlp_workflow"

# This tag is supplied by the build script and will be used to determine the version
# when registering tasks, workflows, and launch plans
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
