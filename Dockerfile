FROM python:3.12-bookworm
LABEL authors="kid"

ENTRYPOINT ["top", "-b"]
