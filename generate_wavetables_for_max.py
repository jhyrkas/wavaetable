import librosa
import math
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import sys

from vae_cqt import vae_cqt
from vae_stft import vae_stft

# will probably add some functions to this so let's make a main
if __name__ == '__main__' :
    gain = 0.75
    z_size = 16
    fs = 16000
    length = 3
    vae = vae_stft()
    vae.load_state_dict(torch.load('vae_stft_model_params.pytorch'))
    vae.eval()
    hop_length = 512
    n_reps = length * int(fs / hop_length)
    data_size = 1025
    i = 0
    while i < 10 :
        z = np.random.normal(size=z_size)
        X_hat = vae.decode(torch.from_numpy(z).float()).detach()
        #X_hat = vae.decode(mu, f0)
        #print(X_hat)

        x_hat = librosa.griffinlim(np.repeat(X_hat.numpy().reshape(data_size,1), n_reps, axis=1))

        x_hat = gain * (x_hat / np.max(np.abs(x_hat)))
        f0_hat_frames, voiced_hat, _ = librosa.pyin(x_hat, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'), sr=fs)
        f0_hat = np.mean(f0_hat_frames[voiced_hat]) if np.sum(voiced_hat) > 10 else 0 # at least 10 voiced frames?
        print(str(f0_hat))
        #print(z)
        if f0_hat == 0 :
            continue
        cycle_samps = 512 # for max's cycle object
        new_fs = math.ceil(cycle_samps * f0_hat)
        new_x_hat = librosa.resample(x_hat, fs, new_fs)
        new_x_hat = new_x_hat / np.max(np.abs(new_x_hat))
        start_index = new_fs//2 # avoid silence at beginning? 
        looping = True
        while looping and start_index < len(new_x_hat):
            if math.isclose(new_x_hat[start_index], 0.0, abs_tol=0.001) :
                looping = False
            else :
                start_index += 1

        if start_index + cycle_samps <= len(new_x_hat) :
            sf.write('output' + str(i) + '.wav', new_x_hat[start_index:start_index+cycle_samps], 44100) # don't think fs matters?
            i += 1
        else :
            # that's an error
            print(new_x_hat[0:cycle_samps])
        
        #sd.play(x, fs)
        #sd.wait()
        #sd.play(x_hat, fs)
        #sd.wait()
