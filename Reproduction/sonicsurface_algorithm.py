"""
Pure-Python port of the SonicSurface amplitude-pattern algorithm.

Original MATLAB: Asier Marzo - "Generating Airborne Ultrasonic Amplitude Patterns
Using an Open Hardware Phased Array", MDPI Applied Sciences 11(7), 2981 (2021).
https://www.mdpi.com/2076-3417/11/7/2981

This implements the iterative Gerchberg-Saxton / angular-spectrum method that
computes the emission phases (and optionally amplitudes) needed on a discretized
phased array in order to recreate a target amplitude image at a parallel slice.
"""

import numpy as np
from scipy import fft
from PIL import Image
import matplotlib.pyplot as plt


def fftasa(p0, z, medium, N, delta, f0):
    """Spectral (angular spectrum) propagator copied from the FOCUS toolbox.

    Propagates the complex field p0 (an [nx,ny] slice) by a distance z.
    """
    soundspeed = medium["soundspeed"]
    attenuationdBcmMHz = medium["attenuationdBcmMHz"]
    wavelen = soundspeed / f0
    dBperNeper = 20.0 * np.log10(np.exp(1.0))
    attenuationNeperspermeter = attenuationdBcmMHz / dBperNeper * 100.0 * f0 / 1e6

    fftpressz0 = fft.fft2(p0, s=(N, N))
    wavenum = 2.0 * np.pi / wavelen

    if N % 2:  # odd
        kx = (np.arange(-N / 2 - 0.5, N / 2 - 0.5 + 1)) * wavelen / (N * delta)
        ky = (np.arange(-N / 2 - 0.5, N / 2 - 0.5 + 1)) * wavelen / (N * delta)
    else:  # even
        kx = (np.arange(-N / 2, N / 2)) * wavelen / (N * delta)
        ky = (np.arange(-N / 2, N / 2)) * wavelen / (N * delta)

    kxspace, kyspace = np.meshgrid(kx, ky)
    kxsq_ysq = fft.fftshift(kxspace ** 2 + kyspace ** 2)
    # complex sqrt: evanescent components (kxsq_ysq>1) yield imaginary kz,
    # exactly like MATLAB's sqrt of a negative number
    kzspace = wavenum * np.sqrt(1.0 - kxsq_ysq + 0j)

    if z > 0:
        H = np.conj(np.exp(1j * z * kzspace))
    else:
        H = np.exp(-1j * z * kzspace) * (kxsq_ysq <= 1.0)

    if attenuationNeperspermeter > 0:
        evans_mode = (np.sqrt(kxsq_ysq) < 1.0).astype(float)
        with np.errstate(invalid="ignore"):
            cos_term = np.cos(np.arcsin(np.sqrt(np.clip(kxsq_ysq, 0, 1))))
        H = H * np.exp(-attenuationNeperspermeter * (z / cos_term) * evans_mode) * evans_mode

    D = (N - 1) * delta
    thres = np.sqrt(0.5 * D ** 2 / (0.5 * D ** 2 + z ** 2))
    filt = (np.sqrt(kxsq_ysq) <= thres).astype(float)
    H = H * filt

    fftpress = fft.ifft2(fftpressz0 * H, s=(N, N))
    return fftpress


def _imresize(arr, new_shape, order):
    """Mimic MATLAB imresize. order=0 -> nearest, order=1 -> bilinear."""
    from scipy.ndimage import zoom
    factors = (new_shape[0] / arr.shape[0], new_shape[1] / arr.shape[1])
    return zoom(arr, factors, order=order, mode="reflect")


def calcEmissionForTargetAmpSlice(targetAmpSlice, dist, iters, sliceSize, freq,
                                  soundSpeed, emitterSize, ampRes, phaseRes):
    """Iteratively compute emission amps/phases for a target amplitude slice.

    Returns amps (1D), phases (1D, rad) and the obtained amplitude slice (2D).
    """
    targetAmpSlice = np.asarray(targetAmpSlice, dtype=complex)
    w, h = targetAmpSlice.shape
    assert w == h
    assert 2 ** int(np.round(np.log2(w))) == w, "side must be a power of 2"

    target = np.zeros((w, h), dtype=complex)
    emission = np.zeros((w, h), dtype=complex)
    nEmittersPerSide = int(np.floor(sliceSize / emitterSize))
    nEmitters = nEmittersPerSide * nEmittersPerSide
    emitterPx = w / nEmittersPerSide

    # amplitude mask: a grid of circles representing the emitters
    mask = np.zeros((w, h))
    emPx2 = emitterPx * emitterPx / 4.0
    ix = np.arange(w) + 1  # MATLAB 1-based
    iy = np.arange(h) + 1
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    diffX = IX - (np.floor((IX - 1) / emitterPx) * emitterPx + emitterPx / 2)
    diffY = IY - (np.floor((IY - 1) / emitterPx) * emitterPx + emitterPx / 2)
    mask = (diffX ** 2 + diffY ** 2 < emPx2).astype(float)

    medium = {"soundspeed": soundSpeed, "attenuationdBcmMHz": 0.0}

    for _ in range(iters):
        # stamp targetAmpSlice into target, keep phases
        target = targetAmpSlice * np.exp(1j * np.angle(target))

        # backpropagate to the emission slice
        emission = fftasa(target, -dist, medium, w, sliceSize / w, freq)

        amp = np.abs(emission)
        phase = np.angle(emission)
        downAmp = _imresize(amp, (nEmittersPerSide, nEmittersPerSide), order=1)
        downPhase = _imresize(phase, (nEmittersPerSide, nEmittersPerSide), order=1)

        if ampRes == 0:
            downAmp = np.ones_like(downAmp)
        else:
            downAmp = np.trunc(downAmp * ampRes) / ampRes
        if phaseRes == 0:
            downPhase = np.zeros_like(downPhase)
        else:
            downPhase = np.trunc(downPhase / np.pi * phaseRes) * np.pi / phaseRes

        amp = _imresize(downAmp, (w, h), order=0)
        phase = _imresize(downPhase, (w, h), order=0)
        amp = amp * mask
        emission = amp * np.exp(1j * phase)

        # propagate forward to the target
        target = fftasa(emission, dist, medium, w, sliceSize / w, freq)

    ampSlice = np.abs(target)

    amps = np.zeros(nEmitters)
    phases = np.zeros(nEmitters)
    index = 0
    for ix in range(1, nEmittersPerSide + 1):
        for iy in range(1, nEmittersPerSide + 1):
            centerX = int(round(ix * emitterPx - emitterPx / 2)) - 1
            centerY = int(round(iy * emitterPx - emitterPx / 2)) - 1
            em = emission[centerX, centerY]
            amps[index] = np.abs(em)
            phases[index] = np.angle(em)
            index += 1

    return amps, phases, ampSlice


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(here, "smiley.png")

    targetAmp = np.array(Image.open(target_file).convert("L"), dtype=float)
    targetAmp = targetAmp / targetAmp.max()

    amps, phases, ampSlice = calcEmissionForTargetAmpSlice(
        targetAmp, dist=0.16, iters=50, sliceSize=0.16,
        freq=40000, soundSpeed=340, emitterSize=0.005,
        ampRes=0, phaseRes=32)

    fig, ax = plt.subplots(2, 1, figsize=(5, 8))
    ax[0].imshow(targetAmp, cmap="viridis")
    ax[0].set_title("target")
    ax[1].imshow(ampSlice, cmap="viridis")
    ax[1].set_title("obtained")
    for a in ax:
        a.axis("off")
    out = os.path.join(here, "python_result.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Saved figure to {out}")

    targetAmp = targetAmp / targetAmp.max()
    ampSlice = ampSlice / ampSlice.max()
    mse = np.sum((targetAmp - ampSlice) ** 2) / ampSlice.size
    print(f"MSE = {mse:.6f}")
    print(f"Computed {len(phases)} emitter phases (range {phases.min():.3f} .. {phases.max():.3f} rad)")
