---
description: Libraries for analyzing music files
---

# Python Libraries

## Librosa

Librosa load audio files including midi, wav files through its load\(\) in librosa.core. This package provides a way of analyzing audio in its raw format. It does not have track information for different tracks of midi file.

The function returns

* y:np.ndarray \[shape=\(n,\) or \(2, n\)\]audio time series
* sr:number &gt; 0 \[scalar\]sampling rate of y

Audio time series is a list of amplitude of the audio, normalized to \(-1, 1\). For instance if the value is 8192 and the file is a 16-bit wav, then this value is normalized to +0.25. Sample rate is something like 44100 for 44100 Hz audio. 



