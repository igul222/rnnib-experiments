import features
import scipy.io.wavfile as wav

(rate,sig) = wav.read("../data/SA2.WAV")
mfcc_feat = features.mfcc(sig,rate)
fbank_feat, fbank_energies = features.fbank(sig,rate)

# print fbank_feat[1:3,:]

for x in fbank_feat:
  print x