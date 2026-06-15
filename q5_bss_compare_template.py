"""Q5 starter: compare FastICA with PCA as a baseline for blind source separation."""
from pathlib import Path
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parent / 'data'
OUT = Path(__file__).resolve().parent / 'q5_outputs'
OUT.mkdir(exist_ok=True)

wavs = sorted(ROOT.rglob('*.wav'))
if len(wavs) != 3:
    raise RuntimeError(f'Expected 3 wav files, found {len(wavs)}')

signals=[]
fs_ref=None
for p in wavs:
    fs, x = wavfile.read(p)
    if fs_ref is None: fs_ref=fs
    if fs != fs_ref: raise RuntimeError('sample rates differ')
    x = x.astype(float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    signals.append(x)
X=np.column_stack(signals)

methods={
    'FastICA': FastICA(n_components=3, whiten='unit-variance', random_state=42, max_iter=3000).fit_transform(X),
    'PCA baseline': PCA(n_components=3, random_state=42).fit_transform(X),
}

def offdiag_abs_corr(Y):
    C=np.corrcoef(Y.T)
    mask=~np.eye(C.shape[0],dtype=bool)
    return float(np.mean(np.abs(C[mask])))

def mean_pairwise_mutual_information(Y, max_samples=5000):
    """Average pairwise MI. Lower values indicate stronger statistical independence."""
    if len(Y) > max_samples:
        idx = np.linspace(0, len(Y)-1, max_samples, dtype=int)
        Z = Y[idx]
    else:
        Z = Y
    values=[]
    for i,j in itertools.combinations(range(Z.shape[1]),2):
        values.append(mutual_info_regression(Z[:,[i]], Z[:,j], random_state=42)[0])
    return float(np.mean(values))

for name,Y in methods.items():
    Y=Y/(Y.std(axis=0,keepdims=True)+1e-12)
    safe_name=name.replace(' ','_')
    print('\n' + name)
    print('  mean absolute off-diagonal correlation =', offdiag_abs_corr(Y))
    print('  mean pairwise mutual information       =', mean_pairwise_mutual_information(Y))
    print('  absolute kurtosis of components        =', np.abs(kurtosis(Y, axis=0, fisher=True)))

    fig=plt.figure(figsize=(9,6))
    for i in range(3):
        ax=fig.add_subplot(3,1,i+1)
        ax.plot(np.arange(min(len(Y),4000))/fs_ref,Y[:4000,i])
        ax.set_ylabel(f's{i+1}')
    ax.set_xlabel('time (s)')
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(OUT / f'q5_{safe_name}.png',dpi=180)
    plt.close(fig)

    for i in range(3):
        audio=Y[:,i]
        audio=audio/(np.max(np.abs(audio))+1e-12)
        wavfile.write(OUT / f'{safe_name}_component_{i+1}.wav', fs_ref, (audio*32767).astype(np.int16))

print(f'\nSaved plots and separated audio files to: {OUT}')
print('Note: correlation alone is not sufficient because PCA decorrelates components by construction.')
print('Use mutual information, non-Gaussianity, spectra, and listening tests for comparison.')
print('Without clean reference sources, exact SDR/SIR cannot be computed.')
