import librosa
import math
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import sys

from pythonosc import osc_message_builder
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from typing import List, Any

from vae_stft import vae_stft

# osc code based on https://python-osc.readthedocs.io/en/latest/dispatcher.html

z = np.zeros(16)
wt = np.zeros(512)
client = SimpleUDPClient("127.0.0.1", 7771)

def update_z(address: str, *args: List[Any]) -> None:
    global z
    if not address[:6] == "/param" : 
        return

    # not doing much error checking here
    i = int(address[6:])
    v = float(args[0])
    z[i] = v

def update_wavetable(vae) :
    global wt
    global z
    gain = 0.75
    z_size = 16
    fs = 16000
    length = 3
    hop_length = 512
    n_reps = length * int(fs / hop_length)
    data_size = 1025
    X_hat = vae.decode(torch.from_numpy(z).float()).detach()
    x_hat = librosa.griffinlim(np.repeat(X_hat.numpy().reshape(data_size,1), n_reps, axis=1))
    x_hat = gain * (x_hat / np.max(np.abs(x_hat)))
    f0_hat_frames, voiced_hat, _ = librosa.pyin(x_hat, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'), sr=fs)
    f0_hat = np.mean(f0_hat_frames[voiced_hat]) if np.sum(voiced_hat) > 10 else 0 # at least 10 voiced frames?
    if f0_hat == 0 :
        print('F0 ESTIMATION FAILED')
        return False # something here...
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
        wt = new_x_hat[start_index:start_index+cycle_samps] # do something with this later
        return True
    else :
        print('ERROR IN WAVETABLE GENERATION')
        return False

def send_wavetable(address: str, fixed_args: List[Any], *osc_args: List[Any]) -> None :
    global client
    global wt
    vae = fixed_args[0]
    if update_wavetable(vae) :
        try :
            tmp = wt.astype(np.float32)
            builder = osc_message_builder.OscMessageBuilder(address="/scRecv")
            builder.add_arg(tmp.tobytes(), builder.ARG_TYPE_BLOB)
            message = builder.build()
            client.send_message("/scRecv", message)
            print('sent wavetable')
        except :
            # had an infinite value once but i missed the exception type or where it occurred...
            client.send_message("/scErr", 0) # not sure if we have to send a "message"

def listen_to_timbre(address: str, fixed_args: List[Any], *osc_args: List[Any]) -> None :
    global wt
    vae = fixed_args[0]
    if update_wavetable(vae) :
        gain = 0.5
        fs = 44100
        sig = np.tile(wt, (3*44100) // len(wt)) * .666
        sd.play(sig, fs)

if __name__ == '__main__' :
    # OSC set up
    dispatcher = Dispatcher()

    # NN set up
    vae = vae_stft()
    vae.load_state_dict(torch.load('vae_stft_model_params.pytorch'))
    vae.eval()

    dispatcher.map("/param*", update_z)
    dispatcher.map("/generate", send_wavetable, vae)
    dispatcher.map("/listen", listen_to_timbre, vae)
    server = BlockingOSCUDPServer(("127.0.0.1", 1337), dispatcher)
    while True :
        server.handle_request()
