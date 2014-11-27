#! /bin/bash

./phoneme_transcription.py training.txt training_raw.nc
normalise_inputs.sh training_raw.nc

./phoneme_transcription.py -m training.txt training_mfc.nc
normalise_inputs.sh training_mfc.nc

./phoneme_transcription.py validation.txt validation_raw.nc
normalise_inputs.sh validation_raw.nc

./phoneme_transcription.py -m validation.txt validation_mfc.nc
normalise_inputs.sh validation_mfc.nc

./phoneme_transcription.py test.txt test_raw.nc
normalise_inputs.sh test_raw.nc

./phoneme_transcription.py -m test.txt test_mfc.nc
normalise_inputs.sh test_mfc.nc