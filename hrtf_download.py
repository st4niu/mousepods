# SPDX-License-Identifier: Apache-2.0
"""First use of spatial audio: fetch the HRTF (how sound reaches each ear from every direction).

MousePods uses the MIT KEMAR dummy-head measurements (Bill Gardner and Keith Martin, MIT Media
Lab) in the SOFA format, as published by the SOFA conventions database. The file is not shipped
with MousePods: it is downloaded once (1.1 MB) from sofacoustics.org, checked against a fixed
SHA-256, and kept in <data dir>/hrtf/.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request

URL = "https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa"
NAME = "mit_kemar_normal_pinna.sofa"
SHA256 = "e7035994f5fd754058424c061380ee92b1d5ed58fccef2887a4266916616acdf"
SIZE = 1171764
MAX_BYTES = 4 * 1024 * 1024


def path_in(data_dir: str) -> str:
    return os.path.join(data_dir, "hrtf", NAME)


def valid(path: str) -> bool:
    try:
        if os.path.getsize(path) != SIZE:
            return False
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest() == SHA256
    except OSError:
        return False


def ensure(data_dir: str, timeout: float = 30.0, opener=urllib.request.urlopen) -> str:
    """Path of a verified HRTF in data_dir, downloading it if needed. Raises OSError/ValueError."""
    dest = path_in(data_dir)
    if valid(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "MousePods"})
    with opener(req, timeout=timeout) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("the downloaded HRTF does not match the expected file")
    fd, tmp = tempfile.mkstemp(prefix=".hrtf-", dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
    return dest
