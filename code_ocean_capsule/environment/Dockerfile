# hash:sha256:098f3c06e23cede38d182b265cd0c7e0b97379e1fcd424ce780593e5c7556c14
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

COPY postInstall /
RUN bash /postInstall
