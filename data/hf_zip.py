"""Range-based access to deflated files inside a remote Hugging Face ZIP."""

from __future__ import annotations

import os
import struct
import tempfile
import urllib.request
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator


@dataclass(frozen=True)
class ZipMember:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int

    @property
    def label(self) -> str:
        parts = PurePosixPath(self.name).parts
        if len(parts) < 3:
            raise ValueError(f"video member has no split and label: {self.name}")
        return parts[-2]

    @property
    def split(self) -> str:
        parts = PurePosixPath(self.name).parts
        if len(parts) < 3:
            raise ValueError(f"video member has no split and label: {self.name}")
        return parts[-3]


class RemoteZipArchive:
    """Index and extract selected ZIP members using HTTP byte ranges."""

    def __init__(self, repo_id: str, filename: str = "dataset_v2.zip", revision: str = "main") -> None:
        self.url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
        self.cache_directory = tempfile.TemporaryDirectory(prefix="sign_speech_videos_")
        self._members = self._read_members()

    def _request(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request) as response:
            data = response.read()
            if response.status != 206:
                raise RuntimeError("Hugging Face did not honor the ZIP byte-range request")
            return data

    def _read_members(self) -> list[ZipMember]:
        head_request = urllib.request.Request(self.url, method="HEAD")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            head_request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(head_request) as response:
            archive_size = int(response.headers["Content-Length"])
        tail_start = max(0, archive_size - 65557)
        tail = self._request(tail_start, archive_size - 1)
        eocd_offset = tail.rfind(b"PK\x05\x06")
        if eocd_offset < 0:
            raise ValueError("remote file is not a ZIP archive")
        _, _, _, _, count, directory_size, directory_offset, comment_size = struct.unpack_from(
            "<4s4H2LH", tail, eocd_offset
        )
        directory = self._request(directory_offset, directory_offset + directory_size - 1)
        members: list[ZipMember] = []
        offset = 0
        for _ in range(count):
            if directory[offset : offset + 4] != b"PK\x01\x02":
                raise ValueError("invalid ZIP central directory")
            fields = struct.unpack_from("<4s6H3L5H2L", directory, offset)
            _, _, _, _, compression, _, _, _, compressed_size, uncompressed_size, name_size, extra_size, comment_size, _, _, _, local_offset = fields
            start = offset + 46
            name = directory[start : start + name_size].decode("utf-8")
            if name.lower().endswith(".mp4"):
                members.append(ZipMember(name, compression, compressed_size, uncompressed_size, local_offset))
            offset += 46 + name_size + extra_size + comment_size
        return members

    def members(self, split: str) -> list[ZipMember]:
        return [member for member in self._members if member.split == split]

    @contextmanager
    def extract(self, member: ZipMember) -> Iterator[str]:
        cached_path = os.path.join(self.cache_directory.name, f"{member.local_header_offset}.mp4")
        if os.path.exists(cached_path):
            yield cached_path
            return
        header = self._request(member.local_header_offset, member.local_header_offset + 29)
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"invalid ZIP local header: {member.name}")
        name_size, extra_size = struct.unpack_from("<HH", header, 26)
        data_start = member.local_header_offset + 30 + name_size + extra_size
        compressed = self._request(data_start, data_start + member.compressed_size - 1)
        if member.compression == 0:
            content = compressed
        elif member.compression == 8:
            content = zlib.decompress(compressed, -15)
        else:
            raise ValueError(f"unsupported ZIP compression method {member.compression}")
        if len(content) != member.uncompressed_size:
            raise ValueError(f"size mismatch while extracting {member.name}")
        with open(cached_path, "wb") as video_file:
            video_file.write(content)
        yield cached_path
