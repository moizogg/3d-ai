"""Minimal ASCII PLY reader/writer for Gaussian inspection.

Supports vertex properties commonly used by splat exports. Binary PLY is
detected and left untouched by the inspector (copy-through).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlyData:
    """In-memory ASCII PLY."""

    comments: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)  # property names in order
    property_types: list[str] = field(default_factory=list)
    vertices: list[list[float]] = field(default_factory=list)
    is_binary: bool = False
    raw_bytes: bytes | None = None

    @property
    def count(self) -> int:
        return len(self.vertices)

    def prop_index(self, name: str) -> int | None:
        try:
            return self.properties.index(name)
        except ValueError:
            return None


def read_ply(path: Path) -> PlyData:
    path = Path(path)
    raw = path.read_bytes()
    # Detect binary
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"Invalid PLY (no end_header): {path}")
    header_text = raw[: header_end + len(b"end_header")].decode("ascii", errors="replace")
    if "format binary" in header_text:
        return PlyData(is_binary=True, raw_bytes=raw, comments=["binary_passthrough"])

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    props: list[str] = []
    types: list[str] = []
    comments: list[str] = []
    vertex_count = 0
    i = 0
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Not a PLY file: {path}")
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line.startswith("comment"):
            comments.append(line[7:].strip())
        elif line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            # property <type> <name>
            if len(parts) >= 3:
                types.append(parts[1])
                props.append(parts[-1])
        elif line == "end_header":
            break

    vertices: list[list[float]] = []
    for _ in range(vertex_count):
        if i >= len(lines):
            break
        parts = lines[i].split()
        i += 1
        row: list[float] = []
        for p in parts[: len(props)]:
            try:
                row.append(float(p))
            except ValueError:
                row.append(0.0)
        # pad
        while len(row) < len(props):
            row.append(0.0)
        vertices.append(row)

    return PlyData(
        comments=comments,
        properties=props,
        property_types=types,
        vertices=vertices,
        is_binary=False,
    )


def write_ply(path: Path, data: PlyData) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.is_binary and data.raw_bytes is not None:
        path.write_bytes(data.raw_bytes)
        return path

    lines = ["ply", "format ascii 1.0"]
    for c in data.comments:
        lines.append(f"comment {c}")
    lines.append(f"element vertex {len(data.vertices)}")
    for t, name in zip(data.property_types, data.properties):
        lines.append(f"property {t} {name}")
    if not data.properties:
        # fallback xyz
        lines.extend(
            [
                "property float x",
                "property float y",
                "property float z",
            ]
        )
        data.properties = ["x", "y", "z"]
        data.property_types = ["float", "float", "float"]
    lines.append("end_header")
    for v in data.vertices:
        lines.append(" ".join(str(x) for x in v[: len(data.properties)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def col_or(data: PlyData, names: list[str], default: float = 0.0) -> list[float]:
    """Extract a property column by first matching name."""
    for name in names:
        idx = data.prop_index(name)
        if idx is not None:
            return [row[idx] if idx < len(row) else default for row in data.vertices]
    return [default] * data.count
