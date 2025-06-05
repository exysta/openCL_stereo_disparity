# Use Ubuntu 24.04 (Noble)
FROM ubuntu:noble

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential  \
    git \
    clinfo \
    cmake  \
    opencl-headers ocl-icd-opencl-dev  \
    intel-opencl-icd  \
    && rm -rf /var/lib/apt/lists/*

# Ensure that the OpenCL libraries and headers are accessible
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Define user variables safely
ARG USERNAME=vscode
ARG USER_UID=2001
ARG USER_GID=2001


# Ensure UID/GID are not in use before adding
RUN if getent group $USER_GID > /dev/null; then echo "Group exists, skipping"; else groupadd --gid $USER_GID $USERNAME; fi \
    && if id -u $USERNAME > /dev/null 2>&1; then echo "User exists, skipping"; else useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; fi


USER $USERNAME
WORKDIR /workspace
