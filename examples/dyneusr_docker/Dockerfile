FROM python:3.7

ADD ../../scripts/dyneusr-cli.py /

RUN pip install --user -r ../../requirements.txt

RUN pip install --user -e ../../ 

ENTRYPOINT ["python", "../../scripts/dyneusr-cli.py init -- --interactive"]
