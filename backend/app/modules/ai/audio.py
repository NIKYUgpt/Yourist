"""Декодирование аудио в PCM 16 кГц моно — общий формат для распознавания речи.

Голосовые сообщения приходят в разных форматах (OGG/Opus из мессенджеров, M4A,
MP3, WAV). Декодируем через PyAV (FFmpeg внутри пакета, ставить отдельно не нужно).
"""

import io
from array import array

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16 бит, моно
MAX_SECONDS = 300


class AudioError(Exception):
    pass


def decode_pcm16(audio: bytes, max_seconds: int = MAX_SECONDS) -> bytes:
    import av  # тяжёлый импорт — только там, где действительно нужен звук

    try:
        with av.open(io.BytesIO(audio)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioError("в файле нет звуковой дорожки")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            out = bytearray()
            for frame in container.decode(stream):
                for chunk in resampler.resample(frame):
                    out += bytes(chunk.planes[0])[: chunk.samples * 2]
                if len(out) > max_seconds * BYTES_PER_SECOND:
                    raise AudioError(f"запись длиннее {max_seconds // 60} минут")
            for chunk in resampler.resample(None):
                out += bytes(chunk.planes[0])[: chunk.samples * 2]
    except av.FFmpegError as exc:
        raise AudioError(f"не удалось прочитать аудио: {exc}") from exc
    if not out:
        raise AudioError("пустая запись")
    return bytes(out)


def _energy(pcm: bytes, start: int, end: int) -> int:
    samples = array("h", pcm[start:end])
    return sum(abs(s) for s in samples[::4])


def split_pcm(pcm: bytes, max_seconds: int, search_seconds: int = 4) -> list[bytes]:
    """Куски не длиннее max_seconds; режем в самом тихом месте последних секунд куска,
    чтобы не разрезать слово посередине."""
    limit = max_seconds * BYTES_PER_SECOND
    window = BYTES_PER_SECOND // 20  # 50 мс, кратно 2 байтам
    parts, pos = [], 0
    while len(pcm) - pos > limit:
        lo = pos + limit - search_seconds * BYTES_PER_SECOND
        cut = min(
            range(lo, pos + limit - window + 1, window),
            key=lambda i: _energy(pcm, i, i + window),
        )
        cut += window // 2 // 2 * 2
        parts.append(pcm[pos:cut])
        pos = cut
    parts.append(pcm[pos:])
    return parts
