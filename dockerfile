FROM archlinux AS builder

RUN mkdir /etc/gnupg && \
    echo "honor-http-proxy" > /etc/gnupg/dirmngr.conf && \
    echo "honor-http-proxy" > /etc/pacman.d/gnupg/dirmngr.conf && \
    pacman-key --init && \
    pacman -Syu --noconfirm && \
    pacman -Sy --noconfirm pyenv base-devel openssl zlib xz

ENV PYTHON_VERSION=3.11.5
RUN pyenv install $PYTHON_VERSION

FROM archlinux

RUN mkdir /etc/gnupg && \
    echo "honor-http-proxy" > /etc/gnupg/dirmngr.conf && \
    echo "honor-http-proxy" > /etc/pacman.d/gnupg/dirmngr.conf && \
    pacman-key --init && \
    pacman-key --recv-key 0706B90D37D9B881 FBA220DFC880C036 --keyserver keyserver.ubuntu.com && \
    pacman-key --lsign-key 0706B90D37D9B881 FBA220DFC880C036 && \
    pacman --noconfirm -U 'https://geo-mirror.chaotic.cx/chaotic-aur/chaotic-'{keyring,mirrorlist}'.pkg.tar.zst' && \
    echo "[multilib]" >> /etc/pacman.conf && \
    echo "Include = /etc/pacman.d/mirrorlist" >> /etc/pacman.conf && \
    echo "[chaotic-aur]" >> /etc/pacman.conf && \
    echo "Include = /etc/pacman.d/chaotic-mirrorlist" >> /etc/pacman.conf && \
    pacman -Syu --noconfirm && \
    pacman -Sy --noconfirm python-cleo-git python-poetry-git pyenv

COPY --from=builder /root/.pyenv/versions/$PYTHON_VERSION /root/.pyenv/versions/$PYTHON_VERSION
RUN pyenv rehash && pyenv global $PYTHON_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR bench

COPY bikipy .
COPY pyproject.toml .
COPY poetry.lock .
COPY analysis.sh .

