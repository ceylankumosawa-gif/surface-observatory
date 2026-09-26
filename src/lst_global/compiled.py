"""Optional exact native evaluator for the pinned F model on the Hetzner host.

This is a compiled copy of its tree decisions, not a different fitted model.
The known binary was checked against sklearn on real cohorts and every split
boundary. Unknown/missing inputs remain the strict adapter's responsibility.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import sys

import numpy as np

BACKEND = 'native-c-exact-f-v1'
LIBRARY_PATH = Path('/opt/lst-pilot/models/global-f-native-v1.so')
LIBRARY_SHA256 = '9e2e04534b51f656bb7965b2ca2da4e5f33605607bf83010fe6f0771b756b07b'
LIBRARY_BYTES = 123944
SOURCE_SHA256 = 'f43f1a5824343c815a52b1f982b2d351eb52365115bb963fe393cde102ba053e'
FITTED_MODEL_SHA256 = 'c955f3a69e393eef29dd95e291d751b73e845a43e7ab2067289c6f1cc394f447'


class NativeF:
    def __init__(self, pipeline, *, model_sha256, library_path=LIBRARY_PATH):
        if model_sha256 != FITTED_MODEL_SHA256:
            raise ValueError('Native F only supports its exact fitted model identity.')
        if sys.platform != 'linux' or platform.machine() != 'x86_64':
            raise ValueError('This verified native F binary requires Linux x86_64.')
        with Path(library_path).open('rb') as stream:
            payload = stream.read(LIBRARY_BYTES + 1)
        if len(payload) != LIBRARY_BYTES or hashlib.sha256(payload).hexdigest() != LIBRARY_SHA256:
            raise ValueError('Native F binary differs from the verified artifact.')
        regressor = pipeline.named_steps['regressor']
        if regressor.n_trees_per_iteration_ != 1 or regressor._loss.__class__.__name__ != 'HalfSquaredError':
            raise ValueError('Native F requires the pinned squared-error tree sum.')
        if len(regressor._predictors) != 150 or tuple(np.flatnonzero(regressor._is_categorical_remapped)) != (0,):
            raise ValueError('Native F preprocessing or tree structure differs.')
        categories = regressor._preprocessor.named_transformers_['encoder'].categories_
        if len(categories) != 1 or not np.array_equal(categories[0], np.arange(13)):
            raise ValueError('Native F internal categorical encoding differs.')
        self.pipeline, self.regressor = pipeline, regressor
        # Load precisely the bytes that passed the hash. Seal the anonymous file
        # so a filesystem path cannot change between verification and loading.
        import fcntl
        fd = os.memfd_create('lst-f-verified', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError('Could not stage verified native F bytes.')
                remaining = remaining[written:]
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            self.library = ctypes.CDLL(f'/proc/self/fd/{fd}')
        finally:
            os.close(fd)
        self.function = self.library.predict
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.function.argtypes = [pointer, ctypes.c_size_t, pointer]
        self.function.restype = None

    def predict(self, frame):
        if not 1 <= len(frame) <= 1048576:
            raise ValueError('Native F requires a nonempty bounded feature batch.')
        outer = self.pipeline.named_steps['features'].transform(frame)
        x = np.ascontiguousarray(self.regressor._preprocess_X(outer, reset=False), dtype=np.float64)
        if x.ndim != 2 or x.shape != (len(frame), 40) or not np.isfinite(x).all():
            raise ValueError('Native F refuses incomplete or unexpected preprocessed inputs.')
        if not np.isin(x[:, 0], np.arange(13)).all():
            raise ValueError('Native F refuses unknown categorical codes.')
        output = np.empty(len(x), dtype=np.float64)
        self.function(x, len(x), output)
        if not np.isfinite(output).all():
            raise ValueError('Native F returned nonfinite predictions.')
        return output

    @property
    def provenance(self):
        return {'name': BACKEND, 'library_sha256': LIBRARY_SHA256,
                'compiled_source_sha256': SOURCE_SHA256,
                'model_changed': False, 'fast_math_enabled': False}
