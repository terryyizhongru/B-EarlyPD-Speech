import os
import re
import sys
import glob
import joblib
import librosa
import argparse
import numpy as np
from pathlib import Path
import shutil
from contextlib import contextmanager
import signal
import threading
import tempfile
import soundfile as sf
if not hasattr(np, 'int'):
    np.int = int
from tqdm import tqdm
from distutils.util import strtobool
import warnings
warnings.filterwarnings("ignore")


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


class ExtractionTimeoutError(TimeoutError):
    pass


@contextmanager
def time_limit(seconds: int, *, error_message: str = 'Timed out'):
    """Raise ExtractionTimeoutError if the block exceeds `seconds`.

    Uses SIGALRM, so it only works reliably on Unix and in the main thread of a process.
    """

    if seconds is None or seconds <= 0:
        yield
        return

    if not hasattr(signal, 'SIGALRM'):
        yield
        return

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise ExtractionTimeoutError(error_message)

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

# Monkey patch for the sys.warn error in DisVoice
def monkey_patch_disvoice():
    """Apply monkey patches to fix issues in DisVoice library"""
    # Add sys.warn as an alias for warnings.warn to fix the GCI module
    if not hasattr(sys, 'warn'):
        sys.warn = warnings.warn
        print("Applied monkey patch: sys.warn -> warnings.warn")

# Apply patches before importing DisVoice
monkey_patch_disvoice()

# Use the system-installed DisVoice (site-packages). Avoid importing a local copy.
# Praat must be available for articulation/formants features.
if shutil.which('praat') is None:
    print("[WARN] 'praat' executable not found in PATH. DisVoice articulation extraction may fail.")

try:
    from disvoice.glottal import Glottal
    from disvoice.prosody import Prosody
    from disvoice.phonation import Phonation
    from disvoice.articulation import Articulation

    # DisVoice writes intermediate files under disvoice/tempfiles/. Ensure it exists.
    import disvoice  # noqa: E402
    print(f"Using system disvoice at: {disvoice.__file__}")
    _disvoice_pkg_dir = Path(disvoice.__file__).resolve().parent
    (_disvoice_pkg_dir / 'tempfiles').mkdir(parents=True, exist_ok=True)
except ModuleNotFoundError as e:
    if e.name == 'disvoice':
        raise ModuleNotFoundError(
            "Could not import 'disvoice' in this Python environment. "
            f"Python: {sys.executable}\n"
            "Activate the environment that has DisVoice installed, or install it (e.g. `pip install disvoice`)."
        ) from e
    raise


def extract_prosody():
    prosody = Prosody()

    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'prosody')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    for wav_file in tqdm(wav_files, desc='Prosody', unit='file'):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')
        if (not args.overwrite) and os.path.exists(output_path):
            continue
        prosody_features = prosody.extract_features_file(wav_file, static=True, plots=False, fmt='npy')
        prosody_features = np.expand_dims(prosody_features, 0) if args.static_features else prosody_features
        # -- data curation
        prosody_features[np.isnan(prosody_features)] = 0
        # print(f'Prosody: {prosody_features.shape}')
        np.savez_compressed(output_path, data=prosody_features)

def extract_articulation():
    articulation = Articulation()

    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'articulation')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    for wav_file in tqdm(wav_files, desc='Articulation', unit='file'):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')
        if (not args.overwrite) and os.path.exists(output_path):
            continue
        try:
            articulation_features = articulation.extract_features_file(wav_file, static=True, plots=False, fmt='npy')

            # -- data curation
            articulation_features[np.isnan(articulation_features)] = 0
            # print(f'Articulation: {articulation_features.shape}')
            np.savez_compressed(output_path, data=articulation_features)
        except Exception:
            with open('failed_files.txt', 'a') as f:
                f.write(f'{wav_file}\n')
            print(f'Failed to extract articulation features for {wav_file}')

def extract_phonation():
    phonation = Phonation()

    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'phonation')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    for wav_file in tqdm(wav_files, desc='Phonation', unit='file'):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')
        if (not args.overwrite) and os.path.exists(output_path):
            continue
        phonation_features = phonation.extract_features_file(wav_file, static=True, plots=False, fmt='npy')

        # -- data curation
        phonation_features[np.isnan(phonation_features)] = 0
        # print(f'Phonation: {phonation_features.shape}')
        np.savez_compressed(output_path, data=phonation_features)

def extract_glottal():
    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'glottal')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    def _process_one(wav_file: str):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')
        if (not args.overwrite) and os.path.exists(output_path):
            return ('skipped', wav_file, None)

        tmp_wav_path = None
        try:
            wav_for_extraction = wav_file

            # Optionally truncate to the first N seconds by writing a temporary wav.
            if getattr(args, 'max_audio_sec', 0) and args.max_audio_sec > 0:
                with sf.SoundFile(wav_file) as f:
                    sr = f.samplerate
                    frames = int(sr * float(args.max_audio_sec))
                    audio = f.read(frames=frames, dtype='float32', always_2d=True)
                # Convert to mono (DisVoice is typically mono-oriented)
                if audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                else:
                    audio = audio[:, 0]

                tmp = tempfile.NamedTemporaryFile(prefix='disvoice_clip_', suffix='.wav', delete=False)
                tmp_wav_path = tmp.name
                tmp.close()
                sf.write(tmp_wav_path, audio, sr)
                wav_for_extraction = tmp_wav_path

            # Create the extractor inside the worker to avoid pickling issues.
            with time_limit(args.timeout_sec, error_message=f'Timed out (> {args.timeout_sec}s)'):
                glottal = Glottal()
                glottal_features = glottal.extract_features_file(wav_for_extraction, static=True, plots=False, fmt='npy')
            glottal_features[np.isnan(glottal_features)] = 0
            np.savez_compressed(output_path, data=glottal_features)
            return ('ok', wav_file, None)
        except Exception as e:
            return ('failed', wav_file, repr(e))
        finally:
            if tmp_wav_path is not None:
                try:
                    os.remove(tmp_wav_path)
                except OSError:
                    pass

    # Parallelize across files (glottal extraction is CPU-heavy).
    if getattr(args, 'num_jobs', 1) and args.num_jobs > 1:
        with tqdm_joblib(tqdm(total=len(wav_files), desc='Glottal', unit='file')):
            results = joblib.Parallel(n_jobs=args.num_jobs, backend='loky')(
                joblib.delayed(_process_one)(wav_file) for wav_file in wav_files
            )
    else:
        results = [_process_one(wav_file) for wav_file in tqdm(wav_files, desc='Glottal', unit='file')]

    failed = [wav_file for status, wav_file, _ in results if status == 'failed']
    if failed:
        with open('failed_files.txt', 'a') as f:
            for wav_file in failed:
                f.write(f'{wav_file}\n')
        print(f'Failed to extract glottal features for {len(failed)}/{len(wav_files)} files (see failed_files.txt)')

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='DisVoice-based Feature Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wav-dir', type=str, default='./data/gita/norm_audios/')
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--static-features', action='store_true', default=True, help='Whether to use static features')
    parser.add_argument('--num-jobs', type=int, default=1, help='Number of parallel workers for per-file extraction')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Recompute features even if output exists')
    parser.add_argument('--timeout-sec', type=int, default=600, help='Per-file extraction timeout (seconds). Use 0 to disable')
    parser.add_argument('--max-audio-sec', type=float, default=0, help='Only use the first N seconds of audio for extraction. Use 0 to disable')
    args = parser.parse_args()

    # -- speech feature extraction
    extract_prosody()
    extract_phonation()
    extract_articulation()

    extract_glottal()
