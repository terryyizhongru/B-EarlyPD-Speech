import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import librosa


def _log2_minmax_to_uint8(power: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    z = np.log2(np.maximum(power, eps))
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if zmax > zmin:
        z01 = (z - zmin) / (zmax - zmin)
    else:
        z01 = np.zeros_like(z, dtype=np.float32)
    return (np.clip(z01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _viridis_rgb_from_uint8(gray_u8: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm

    z01 = gray_u8.astype(np.float32) / 255.0
    rgba = cm.get_cmap("viridis", 256)(z01)  # (H, W, 4)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def audio_to_spec_image(
    audio_path: Path,
    out_path: Path,
    *,
    spec_type: str = "standard",  # "standard" | "mel"
    target_size=(600, 600),
    sr=16000,
    max_length_s=10.0,  # seconds; 不足补 0，超过截断
    n_fft=400,
    hop_length=320,
    win_length=400,
    n_mels=80,
    fmin=0.0,
    fmax=None,
    top_db=80.0,
    jpg_quality: int = 100,
    dpi: int = 300,
):
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    target_len = int(sr * max_length_s)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    if spec_type == "mel":
        # Align with R pipeline: use torchaudio mel implementation.
        try:
            import torch
            import torchaudio
        except Exception as e:
            raise RuntimeError(
                "torchaudio/torch are required for --type mel to align with the R pipeline. "
                "Install with: pip install torch torchaudio"
            ) from e

        waveform = torch.from_numpy(y).to(torch.float32).unsqueeze(0)  # (1, T)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sr),
            n_fft=int(n_fft),
            win_length=int(win_length),
            hop_length=int(hop_length),
            f_min=float(fmin),
            f_max=float(fmax if fmax is not None else (sr / 2.0)),
            pad=0,
            n_mels=int(n_mels),
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
        )
        S_power = mel_transform(waveform)[0].cpu().numpy()  # (n_mels, frames)
    elif spec_type == "standard":
        D = librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=True,
        )
        S_power = (np.abs(D) ** 2)
    else:
        raise ValueError(f"Unsupported spec_type: {spec_type!r} (expected 'standard' or 'mel')")

    # Align with R pipeline:
    # - log2(power) scale
    # - transpose (R uses $t())
    # - flip vertically (R image() places first row at bottom)
    S_power = S_power.T
    S_power = np.flipud(S_power)

    img_u8 = _log2_minmax_to_uint8(S_power)
    rgb_arr = _viridis_rgb_from_uint8(img_u8)
    rgb = Image.fromarray(rgb_arr, mode="RGB").resize(target_size, resample=Image.BILINEAR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, quality=int(jpg_quality), dpi=(int(dpi), int(dpi)))


def iter_audio_files(root: Path):
    exts = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".ogg"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Convert audio files under class subfolders to spectrogram JPG images for image-based models. "
            "Defaults are mel to 16kHz sampling rate and 10s max length (pad/truncate)."
        )
    )
    ap.add_argument("--in_dir", required=True, help="输入音频根目录（内部按类别分子目录）")
    ap.add_argument("--out_dir", required=True, help="输出图片根目录（会保持同样的类别子目录结构）")
    ap.add_argument("--type", choices=("standard", "mel"), default="mel")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max_length_s", type=float, default=10.0)
    ap.add_argument("--n_fft", type=int, default=400)
    ap.add_argument("--hop_length", type=int, default=320)
    ap.add_argument("--win_length", type=int, default=400)
    ap.add_argument("--n_mels", type=int, default=80, help="仅在 --type mel 时使用")
    ap.add_argument("--fmin", type=float, default=0.0, help="仅在 --type mel 时使用")
    ap.add_argument("--fmax", type=float, default=None, help="仅在 --type mel 时使用；默认 sr/2")
    ap.add_argument(
        "--top_db",
        type=float,
        default=80.0,
        help="Deprecated (kept for compatibility); current output aligns to R log2 scaling and ignores top_db.",
    )
    ap.add_argument("--jpg_quality", type=int, default=100)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--size", type=int, default=600)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    # 假设结构类似：
    # in_dir/
    #   healthy/*.wav
    #   parkinson/*.wav
    audio_files = list(iter_audio_files(in_dir))
    try:
        from tqdm import tqdm

        iterator = tqdm(audio_files, desc="audio2image", unit="file")
    except Exception:
        iterator = audio_files

    failures = 0
    for audio_path in iterator:
        rel = audio_path.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".jpg")
        try:
            audio_to_spec_image(
                audio_path,
                out_path,
                spec_type=args.type,
                target_size=(args.size, args.size),
                sr=args.sr,
                max_length_s=args.max_length_s,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
                top_db=args.top_db,
                jpg_quality=args.jpg_quality,
                dpi=args.dpi,
            )
            print(f"[OK] {audio_path} -> {out_path}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {audio_path}: {e}")

    if failures:
        raise SystemExit(f"Failed to convert {failures} of {len(audio_files)} audio files")


if __name__ == "__main__":
    main()
