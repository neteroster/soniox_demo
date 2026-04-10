"""
main_direct.py — Windows 直接音频捕获 -> Soniox Realtime STT（内置流式翻译） -> Rich 终端

音频捕获模式：
  --mode system   (默认) PyAudioWPatch WASAPI loopback 捕获系统全局音频
  --mode process  --pid <PID> | --name <名称>  ProcTap 捕获指定进程（树）音频
"""

import argparse
import asyncio
import os
import threading

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.text import Text
from soniox import AsyncSonioxClient
from soniox.realtime import RealtimeSTTConfig
from soniox.types.api import TranslationConfig

load_dotenv()

# ── 配置 ─────────────────────────────────────────────────────────────────────

SONIOX_API_KEY = os.environ["SONIOX_API_KEY"]
SONIOX_MODEL = "stt-rt-v4"
TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE", "zh")
SAMPLE_RATE = 16_000
RICH_FPS = 20

console = Console()


# ── 音频捕获：WASAPI loopback（系统全局） ────────────────────────────────────

class SystemAudioCapture:
    def __init__(self, device_index: int | None = None):
        self.device_index = device_index
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    async def start(self, queue: asyncio.Queue[bytes | None]):
        loop = asyncio.get_running_loop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(loop, queue), daemon=True
        )
        self._thread.start()

    def _run(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[bytes | None]):
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            if self.device_index is not None:
                dev = pa.get_device_info_by_index(self.device_index)
            else:
                wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
                dev = speakers
                for i in range(pa.get_device_count()):
                    d = pa.get_device_info_by_index(i)
                    if d.get("isLoopbackDevice") and d["name"].startswith(
                        speakers["name"].split(" (")[0]
                    ):
                        dev = d
                        break

            rate = int(dev["defaultSampleRate"])
            ch = int(dev["maxInputChannels"])
            console.log(f"[bold green]WASAPI loopback: {dev['name']} ({rate}Hz, {ch}ch)")

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=ch,
                rate=rate,
                input=True,
                input_device_index=int(dev["index"]),
                frames_per_buffer=1024,
            )
            ratio = rate // SAMPLE_RATE

            while not self._stop.is_set():
                data = stream.read(1024, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                if ch >= 2:
                    samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)
                if ratio > 1:
                    samples = samples[::ratio]
                loop.call_soon_threadsafe(queue.put_nowait, samples.tobytes())

            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


# ── 音频捕获：ProcTap（进程级） ──────────────────────────────────────────────

class ProcessCapture:
    def __init__(self, pid: int):
        self.pid = pid
        self._task: asyncio.Task | None = None

    async def start(self, queue: asyncio.Queue[bytes | None]):
        self._task = asyncio.create_task(self._run(queue))

    async def _run(self, queue: asyncio.Queue[bytes | None]):
        from proctap import ProcessAudioCapture

        console.log(f"[bold green]ProcTap: capturing PID {self.pid}")
        tap = ProcessAudioCapture(pid=self.pid)
        tap.start()
        try:
            async for chunk in tap.iter_chunks():
                samples = np.frombuffer(chunk, dtype=np.float32)
                if samples.shape[0] % 2 == 0:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                samples = samples[::3]  # 48kHz -> 16kHz
                samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
                await queue.put(samples.tobytes())
        except Exception as e:
            console.log(f"[bold red]ProcTap error: {e}")
        finally:
            tap.stop()
            await queue.put(None)

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ── 数据 & 渲染 ─────────────────────────────────────────────────────────────

class Segment:
    __slots__ = ("src", "translated", "final")

    def __init__(self, text: str = "", translated: str = "", *, final: bool = False):
        self.src = text
        self.translated = translated
        self.final = final


segments: list[Segment] = []


def render_view() -> Text:
    parts: list[Text] = []
    for seg in segments[-8:]:
        src_style = "bold white" if seg.final else "italic grey50"
        tgt_style = "bold cyan" if seg.final else "italic cyan"
        parts.append(Text("* " + seg.src.replace("\n", " "), style=src_style))
        if seg.translated:
            parts.append(Text("  -> " + seg.translated.replace("\n", " "), style=tgt_style))
    return Text("\n").join(parts)


# ── 进程名 -> PID 选择 ──────────────────────────────────────────────────────

def _resolve_pid(name: str) -> int:
    import psutil

    matches: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if name.lower() in proc.info["name"].lower():
                cmd = " ".join(proc.info["cmdline"] or [])
                matches.append((proc.info["pid"], (cmd[:120] if cmd else proc.info["name"])))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not matches:
        console.print(f"[bold red]No process matching '{name}'")
        raise SystemExit(1)

    if len(matches) == 1:
        console.log(f"[bold green]Found: PID {matches[0][0]} — {matches[0][1]}")
        return matches[0][0]

    console.print(f"\n[bold yellow]Found {len(matches)} processes matching '{name}':\n")
    for i, (pid, label) in enumerate(matches, 1):
        console.print(f"  [cyan]{i:>3}[/cyan]  PID {pid:<8} {label}")
    console.print()

    while True:
        choice = console.input("[bold]Enter index (or PID): [/bold]").strip()
        if not choice:
            continue
        try:
            val = int(choice)
        except ValueError:
            continue
        if 1 <= val <= len(matches):
            return matches[val - 1][0]
        if any(pid == val for pid, _ in matches):
            return val
        console.print("[red]Invalid, try again[/red]")


# ── 主函数 ───────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Windows audio capture + Soniox STT")
    parser.add_argument("--mode", choices=["system", "process"], default="system")
    parser.add_argument("--pid", type=int, help="Target PID (process mode)")
    parser.add_argument("--name", type=str, help="Target process name (process mode)")
    parser.add_argument("--device", type=int, help="WASAPI device index (system mode)")
    args = parser.parse_args()

    if args.mode == "process" and args.pid is None and args.name is None:
        parser.error("--mode process requires --pid or --name")
    if args.mode == "process" and args.pid is None:
        args.pid = _resolve_pid(args.name)

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    if args.mode == "process":
        capture = ProcessCapture(args.pid)
    else:
        capture = SystemAudioCapture(device_index=args.device)
    await capture.start(audio_queue)

    client = AsyncSonioxClient(api_key=SONIOX_API_KEY)
    config = RealtimeSTTConfig(
        model=SONIOX_MODEL,
        audio_format="s16le",
        num_channels=1,
        sample_rate=SAMPLE_RATE,
        enable_endpoint_detection=True,
        translation=TranslationConfig(type="one_way", target_language=TARGET_LANGUAGE),
    )

    console.log("[bold green]Listening … Ctrl+C to stop")

    try:
        async with client.realtime.stt.connect(config=config) as session:

            async def sender():
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        await session.finish()
                        break
                    await session.send_byte_chunk(chunk)

            sender_task = asyncio.create_task(sender())

            final_src = ""
            final_tgt = ""

            with Live(render_view(), refresh_per_second=RICH_FPS, console=console) as live:
                async for event in session.receive_events():
                    nonfinal_src = ""
                    nonfinal_tgt = ""
                    has_end = False

                    for tok in event.tokens:
                        if tok.text == "<end>":
                            has_end = True
                            continue
                        is_tl = tok.translation_status == "translation"
                        if tok.is_final:
                            if is_tl:
                                final_tgt += tok.text
                            else:
                                final_src += tok.text
                        else:
                            if is_tl:
                                nonfinal_tgt += tok.text
                            else:
                                nonfinal_src += tok.text

                    if has_end:
                        if segments and not segments[-1].final:
                            segments[-1].src = final_src
                            segments[-1].translated = final_tgt
                            segments[-1].final = True
                        elif final_src:
                            segments.append(Segment(final_src, final_tgt, final=True))
                        final_src = ""
                        final_tgt = ""
                        live.update(render_view())
                        continue

                    disp_src = final_src + nonfinal_src
                    disp_tgt = final_tgt + nonfinal_tgt
                    if disp_src or disp_tgt:
                        if not segments or segments[-1].final:
                            segments.append(Segment(disp_src, disp_tgt))
                        else:
                            segments[-1].src = disp_src
                            segments[-1].translated = disp_tgt
                        live.update(render_view())

                await sender_task

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.log("[bold yellow]Stopping …")

    await capture.stop()


if __name__ == "__main__":
    asyncio.run(main())
