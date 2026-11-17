ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.6:2.0.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    DEVICE_TARGET=GPU

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["/bin/bash"]
