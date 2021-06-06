# wavaetable
This repo contains Python, Max and SuperCollider code for the WaVAEtable Synthesis method.

## Dependencies

### Python

The python code used here uses numpy, librosa, sounddevice, soundfile, and torch.

### Max and SuperCollider

The Max patches and SuperCollider script do not use any external dependencies.

## Using the Software

### Max

To use the Max controller, first run the script 

>> python generate_wavetables_for_max.py

Then, open nn_osc_controller.maxpat in Max and use a MIDI controller to send MIDI notes to the patch.
You can use the number boxes to assign wavetables (0-9) to a given voice (1-5), or all voices (0). 
You can also use MIDI CC Controls 1-5 to assign wavetables to voices 1-5 respectively, CC 6 to assign one wavetable to all voices, 
and CC 7 and 8 to change the frequency modulation ratio and index of the playback.


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

### Using alternate generative models

One goal of this method is the ability to use other timbral generative models as wavetable generators. One such example is the CANNe synthesizer (https://ee.cooper.edu/~keene/assets/dafx2018_submission42_revised_jcolonel.pdf). To use CANNe with the SuperCollider script, clone this repo: https://github.com/jhyrkas/canne_synth (this is forked from the original CANNe repo, associated with my Network Modulation Synthesis project presented at ICMC 2021). 

After downloading the repo, copy nn_osc_controller.py from the subdirectory probably_useful_for_author_only/ into the main directory (i.e. alongside canne.py). Run nn_osc_controller.py as before to use the CANNe synthesizer with SuperCollider. Note that the CANNe model only uses the first 8 latent parameters (the first two rows). However, it sounds better than the model provided in this repo in my opinion, and nicely compliments the SuperCollider GUI and etude provided here.
