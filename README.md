# wavaetable
Contains Python, Max and SuperCollider code for WaVAEtable Synthesis method

## Dependencies

### Python

The python code used here uses numpy, librosa, sounddevice, soundfile, and torch.

### Max and SuperCollider

The Max patches and SuperCollider script do not use any external dependencies.

## Using the Software

### Max

To use the Max controller, first run the script 

>> python generate_wavetables_for_max.py

Then, open nn_osc_controller.maxpat in Max. Use the number boxes to assign wavetables (0-9) to a given voice (1-5), or all voices (0). Use a MIDI controller to send MIDI notes to the patch, and you can use MIDI CC Controls 1 and 2 to change the frequency modulation ratio and index of the playback.

### SuperCollider

Boot SuperCollider and run the first portion of the script nn_osc_controller.scd that creates the GUI. Also, from the terminal run the script

>> python nn_osc_controller.py

Use the GUI to control the latent parameters, listen to the wavetable (audio will be played from the Python script) and store the wavetable if you would like to. You can use Osc and VOsc UGens in SuperCollider to play the wavetables.

If you would like, after storing 5 wavetables of interest, run the SynthDef command in nn_osc_controller.scd and then run the etude so hear a mini composition using wavetable modulation and interpolation.

### Training the Python Model

If you would like to train the model from scratch, perhaps changing the training parameter or modifying the training data, first download the data (WARNING: pretty large) by running

>> cd data/
>> 
>> sh get_data.sh
>> 
>> python create_data.py

Then:

>> cd ..
>> 
>> python vae_stft.py

Although it is still a work in progress, you can use the \_cqt.py versions of these scripts to train a model that uses the Constant-Q Transform instead of the Fourier Transform. Audio quality is still an issue in this method and will be explored further in the future.
