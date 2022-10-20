FROM python:3.8-slim
LABEL author.name="Jovinder Singh" author.email="jovinder.singh@yahoo.com"
ENV DASH_DEBUG_MODE True

# set the working directory
RUN mkdir /app
WORKDIR /app

# copy the requirements.txt file
COPY requirements.txt requirements.txt

# install all the libraries listed as requirements
RUN pip install -r requirements.txt

COPY ./ ./

# set the port to listen to
EXPOSE 8050

CMD gunicorn --bind 0.0.0.0:8050 main:server
