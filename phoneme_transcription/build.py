#!/usr/bin/python

import netcdf_helpers
# import re
# from scipy import *
# import array
# import wave
# import operator
# from optparse import OptionParser
# import sys
import scipy.io.wavfile as wav
import features
import subprocess

NUM_PHONES = 39 # 39, 48, or null = all

def generateLabels(numPhones):
  labelDict = {label: label for label in ['h#','b','d','g','p','t','k','dx','q',
    'jh','ch','s','sh','z','zh', 'f','th','v','dh','m','n','ng','em','en','eng',
    'nx','l','r','w','y','hh', 'hv','el','iy','ih','eh','ey','ae','aa','aw',
    'ay','ah','ao','oy','ow','uh', 'uw','ux','er','ax','ix','axr','ax-h','pau',
    'epi','bcl','dcl','gcl','pcl', 'kcl','tcl'] }

  if numPhones == 39:
    reducedLabels = {
      'h#': 'sil',
      'pau': 'sil',
      'epi': 'sil',
      'bcl': 'sil',
      'dcl': 'sil',
      'gcl': 'sil',
      'kcl': 'sil',
      'pcl': 'sil',
      'tcl': 'sil',
      'hv': 'hh',
      'eng': 'ng',
      'nx': 'n',
      'en': 'n',
      'em': 'm',
      'axr': 'er',
      'ux': 'uw',
      'el': 'l',
      'zh': 'sh',
      'aa': 'ao',
      'ix': 'ih',
      'ax': 'ah',
      'ax-h': 'ah',
      'q': 'NULL'
    }
  elif numPhones == 48:
    reducedLabels = {
      'h#': 'sil',
      'pau': 'sil',
      'bcl': 'vcl',
      'dcl': 'vcl',
      'gcl': 'vcl',
      'kcl': 'cl',
      'pcl': 'cl',
      'tcl': 'cl',
      'hv': 'hh',
      'eng': 'ng',
      'nx': 'n',
      'em': 'm',
      'axr': 'er',
      'ux': 'uw',
      'ax-h': 'ah',
      'q': 'NULL'
    }
  else:
    reducedLabels = {}

  # labelDict = labelDict union reducedLabels
  labelDict = dict(labelDict.items() + reducedLabels.items())

  labels = set(labelDict.values())
  labels.discard('NULL')

  return (labels, labelDict)

def generateDeltas(values):
  deltas = []
  for t1, frame in enumerate(values):
    deltas.append([])
    for i, value in enumerate(frame):
      num = 0.0
      den = 0.0
      for t2 in range(max(0, t1 - 2), min(t1 + 3, len(values))):
        num += (t2 - t1)*(values[t2][i] - value)
        den += (t2 - t1)*(t2 - t1)
      deltas[t1].append(num / den)
  return deltas

labels, labelDict = generateLabels(NUM_PHONES)

for group in ['training.txt', 'validation.txt', 'test.txt']:
  sequences = file(group).read().strip().split('\n')

  inputLengths = [] # Length of each input sequence
  inputs = [] # Flat array of all input values in all frames of all sequences
  targetStrings = [] # Target transcription string of each sequence

  for wavFilename in sequences:
    (rate, sig) = wav.read(wavFilename)
    mfcc = features.mfcc(sig,rate)
    # fbank_feat, fbank_energies = features.fbank(sig,rate)
    deltas = generateDeltas(mfcc)
    deltaDeltas = generateDeltas(deltas)
    featureVector = [mfcc[i] + deltas[i] + deltaDeltas[i] for i in range(len(mfcc))]
    inputs.extend(featureVector)
    inputLengths.append(len(featureVector))

    phnFilename = wavFilename.replace('.WAV','.PHN')
    phoneLines = file(phnFilename).read().strip().split('\n')
    phones = [line.split()[2] for line in phoneLines]

    targetLabels = [labelDict[phone] for phone in phones if labelDict[phone] != 'NULL']
    targetStrings.append(' '.join(targetLabels))

  # create a new .nc file
  ncFilename = group.replace('.txt', '.nc')
  ncFile = netcdf_helpers.NetCDFFile(ncFilename, 'w')

  # For descriptions of what the dimensions/variables mean to RNNLIB, see
  # http://sourceforge.net/p/rnnl/wiki/Home/#data-file-format

  # Dimensions: 
  netcdf_helpers.createNcDim(ncFile, 'numSeqs',       len(sequences))
  netcdf_helpers.createNcDim(ncFile, 'numTimesteps',  sum(inputLengths))
  netcdf_helpers.createNcDim(ncFile, 'inputPattSize', len(inputs[0]))
  netcdf_helpers.createNcDim(ncFile, 'numDims',       1)
  netcdf_helpers.createNcDim(ncFile, 'numLabels',     len(labels))

  # Variables:
  seqDims = [[n] for n in inputLengths] # I have no clue what this is.
  netcdf_helpers.createNcStrings(ncFile, 'seqTags',       sequences,     ('numSeqs','maxSeqTagLength'),     'sequence tags')
  netcdf_helpers.createNcStrings(ncFile, 'labels',        labels,        ('numLabels','maxLabelLength'),    'labels')
  netcdf_helpers.createNcStrings(ncFile, 'targetStrings', targetStrings, ('numSeqs','maxTargStringLength'), 'target strings')
  netcdf_helpers.createNcVar(ncFile, 'seqLengths', inputLengths, 'i', ('numSeqs',),                     'sequence lengths')
  netcdf_helpers.createNcVar(ncFile, 'seqDims',    seqDims,      'i', ('numSeqs','numDims'),            'sequence dimensions')
  netcdf_helpers.createNcVar(ncFile, 'inputs',     inputs,       'f', ('numTimesteps','inputPattSize'), 'input patterns')

  ncFile.close()

  subprocess.call('normalise_inputs.sh ' + ncFilename, shell=True)