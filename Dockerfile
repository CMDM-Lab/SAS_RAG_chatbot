FROM python:3.12-slim
RUN apt-get update && apt-get install -y build-essential curl
#RUN curl -fsSL https://ollama.com/install.sh | sh
ARG CHATBOT_ROOT=/usr/local/chatbot
WORKDIR $CHATBOT_ROOT
COPY chatbot_requirement.txt /requirement.txt
RUN pip install --no-cache-dir -r /requirement.txt
COPY ./chatbot $CHATBOT_ROOT
#https://github.com/ollama/ollama/issues/546
COPY chatbot_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ARG USER=ollama
ARG GROUP=ollama
#RUN addgroup $GROUP && adduser -D -G $GROUP $USER
USER $USER:$GROUP

ENTRYPOINT ["/entrypoint.sh"]