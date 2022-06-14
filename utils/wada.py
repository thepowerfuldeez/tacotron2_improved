#!/usr/bin/python
# -*- coding: utf-8 -*-

"""WADA external sub modul
"""

# goal : Signal to Noise Ratio estimation with WADA technique

# Metrics created by Chanwoo Kim, Richard M. Stern from paper : Robust Signal-to-Noise Ratio Estimation Based on
# Waveform Amplitude Distribution Analysis

# original matlab code come from : https://labrosa.ee.columbia.edu/projects/snreval/ by Dan Ellis
# author of python version :  Sebastien Ferreira


import argparse
import os
import logging
import numpy
import sys
import subprocess
import re

table = ([-20, 0.409747739], [-19, 0.409869263], [-18, 0.409985656], [-17, 0.409690892], [-16, 0.409861864],
         [-15, 0.409990055], [-14, 0.410271377], [-13, 0.410526266], [-12, 0.411010238],
         [-11, 0.411432644], [-10, 0.412317178], [-9, 0.413372716], [-8, 0.415264259], [-7, 0.417819198],
         [-6, 0.420772515], [-5, 0.424527992], [-4, 0.429188858], [-3, 0.435103734], [-2, 0.442341951],
         [-1, 0.451614855], [0, 0.462211529], [1, 0.474916474], [2, 0.488838093], [3, 0.505092356],
         [4, 0.523537093], [5, 0.543720882], [6, 0.565324274], [7, 0.588475317], [8, 0.613462118],
         [9, 0.639544959], [10, 0.667508177], [11, 0.695837243], [12, 0.724547622], [13, 0.754147993],
         [14, 0.783231484], [15, 0.81240985], [16, 0.842197752], [17, 0.871664058], [18, 0.900305039],
         [19, 0.928804177], [20, 0.95655449], [21, 0.983534905], [22, 1.010471548], [23, 1.0362095],
         [24, 1.061364248], [25, 1.085793118], [26, 1.109481904], [27, 1.132779949], [28, 1.154728256],
         [29, 1.176273084], [30, 1.197035028], [31, 1.216716938], [32, 1.235358982], [33, 1.253643127],
         [34, 1.271038908], [35, 1.287180295], [36, 1.303028647], [37, 1.318395272], [38, 1.332948173],
         [39, 1.347009353], [40, 1.360572696], [41, 1.373455135], [42, 1.385771224], [43, 1.397335037],
         [44, 1.408563968], [45, 1.41959619], [46, 1.42983624], [47, 1.439584667], [48, 1.449021764],
         [49, 1.458048307], [50, 1.466695685], [51, 1.474869384], [52, 1.48269965], [53, 1.490343394],
         [54, 1.49748214], [55, 1.504351061], [56, 1.510764265], [57, 1.516989146], [58, 1.522909703],
         [59, 1.528578001], [60, 1.533898351], [61, 1.539121095], [62, 1.543906502], [63, 1.54858517],
         [64, 1.553107762], [65, 1.557443906], [66, 1.561649273], [67, 1.565663481], [68, 1.569386712],
         [69, 1.573077668], [70, 1.576547638], [71, 1.57980083], [72, 1.583041292], [73, 1.586024961],
         [74, 1.588806813], [75, 1.591624771], [76, 1.594196895], [77, 1.596931549], [78, 1.599446005],
         [79, 1.601850111], [80, 1.604086681], [81, 1.60627134], [82, 1.608261987], [83, 1.610045475],
         [84, 1.611924722], [85, 1.61369656], [86, 1.615340743], [87, 1.616889049], [88, 1.618389159],
         [89, 1.619853744], [90, 1.621358779], [91, 1.622681189], [92, 1.623904229], [93, 1.625131432],
         [94, 1.626324628], [95, 1.6274027], [96, 1.628427675], [97, 1.629455321], [98, 1.6303307],
         [99, 1.631280263], [100, 1.632041021])


def pcm2float(samples, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    samples : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    """
    if samples.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = numpy.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")
    i = numpy.iinfo(samples.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (samples.astype(dtype) - offset) / abs_max


def get_duration(path):
    """
    Calcul la durée du fichier audio sans avoir a charger le waveform.
    Car pour les fichiers audio de plus de deux heure c'est trop gourmant en mémoire
    :param path: le chemin du fichier audio
    :return: un float pour la durée en seconde
    """
    # valid for any audio file accepted by ffprobe
    args = ("ffprobe", "-show_entries", "format=duration", "-i", path)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = popen.communicate()
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(output))
    return float(match.group())


def read_audio_ffmpeg_from_start_to_duration(path, start, duration):
    """
    Lie un fichier audio et retourne un waveform de la seconde 'start' pendant 'duration' seconde.
    L'avantage est de ne pas charger en mémoire le fichier audio en entier.
    Le fichier de sortie est en 16bit, 16kHz, Mono. Avec pcm2float il est normalisé entre -1 et 1.
    :param path: le chemin du fichier audio
    :param start: début de lecture en seconde
    :param duration: durée de lecture en seconde
    :return: le waveform (numpy) de l'audio en 16bit 16kHz mono et amplitude de (-1 à 1)
    """
    command = ['ffmpeg',
               '-i', path,
               '-f', 's16le',
               '-acodec', 'pcm_s16le',
               '-ss', str(start),
               '-t', str(duration),
               '-ar', '16000',  # ouput will have 44100 Hz
               '-ac', '1',  # stereo (set to '1' for mono)
               '-']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8)
    raw_audio = pipe.stdout.read(int(16000 * 2 * duration))
    audio_array = numpy.fromstring(raw_audio, dtype="int16")
    return pcm2float(audio_array)


def compute_snr(path, start, duration):
    """
    Calcul le snr (version WADA) du fichier audio de la seconde 'start' pendant 'duration' seconde.
    :param path: chemin du fichier audio
    :param start: début de lecture en seconde
    :param duration: durée de lecture en seconde
    :return: le snr (float) calculé avec la méthode WADA de la seconde start pendant duration secondes.
    """
    if (isinstance(duration, float) or isinstance(duration, int)) and \
            (isinstance(start, float) or isinstance(start, int)):
        # si duration : float ou int -> une durée en sec ET start float ou int > départ en sec
        try:
            audio = read_audio_ffmpeg_from_start_to_duration(path, start, duration)
        except:
            print('unable to load %s', path)
    else:
        print('PROBLEM : start OU duration dans WADA.py a un type non float ou int')
    D = audio
    aD = numpy.abs(D)
    # Correction eviter problème de log(0)
    aD[aD < 1e-10] = 1e-10
    dVal1 = numpy.mean(aD)
    # Correction eviter problème de log(0)
    if dVal1 < 1e-20:
        dVal1 = 1e-20
    dVal2 = numpy.mean(numpy.log(aD))
    dEng = sum(D ** 2)
    dVal3 = numpy.log(dVal1) - dVal2
    # Correction des Soucis possible si dVal3 prend valeur hors table
    if dVal3 < 0.409747739:
        dVal3 = table[1][1]
    elif dVal3 > 1.6274027:
        dVal3 = table[-6][1]
    # Forcément une valeur grâce aux étapes précédentes
    dSNRix = numpy.max([i for i in range(0, len(table)) if table[i][1] < dVal3])
    dSNR = table[dSNRix][0] + (dVal3 - table[int(dSNRix)][1]) / (table[int(dSNRix) + 1][1] - table[int(dSNRix)][1]) \
           * (table[int(dSNRix) + 1][0] - table[int(dSNRix)][0])
    min_val = float(1e-10)
    # Calculate SNR
    dFactor = 10 ** (dSNR / 10)
    dNoiseEng = dEng / (1 + dFactor)
    dSigEng = dEng * dFactor / (1 + dFactor)
    snr = 10 * numpy.log10(numpy.maximum(numpy.maximum(dSigEng, min_val) / numpy.maximum(dNoiseEng, min_val),
                                         min_val))
    # print('The computed SNR value is ', dSNR)
    # print('Signal energy in this block : ', dSigEng)
    # print('Noise energy in this block : ', dNoiseEng)
    # print('Computed SNR : ', snr)
    return snr


def compute_snr_on_signal(signal):
    D = signal
    aD = numpy.abs(D)
    # Correction eviter problème de log(0)
    aD[aD < 1e-10] = 1e-10
    dVal1 = numpy.mean(aD)
    # Correction eviter problème de log(0)
    if dVal1 < 1e-20:
        dVal1 = 1e-20
    dVal2 = numpy.mean(numpy.log(aD))
    dEng = sum(D ** 2)
    dVal3 = numpy.log(dVal1) - dVal2
    # Correction des Soucis possible si dVal3 prend valeur hors table
    if dVal3 < 0.409747739:
        dVal3 = table[1][1]
    elif dVal3 > 1.6274027:
        dVal3 = table[-6][1]
    # Forcément une valeur grâce aux étapes précédentes
    dSNRix = numpy.max([i for i in range(0, len(table)) if table[i][1] < dVal3])
    dSNR = table[dSNRix][0] + (dVal3 - table[int(dSNRix)][1]) / (table[int(dSNRix) + 1][1] - table[int(dSNRix)][1]) \
           * (table[int(dSNRix) + 1][0] - table[int(dSNRix)][0])
    min_val = float(1e-10)
    # Calculate SNR
    dFactor = 10 ** (dSNR / 10)
    dNoiseEng = dEng / (1 + dFactor)
    dSigEng = dEng * dFactor / (1 + dFactor)
    snr = 10 * numpy.log10(numpy.maximum(numpy.maximum(dSigEng, min_val) / numpy.maximum(dNoiseEng, min_val),
                                         min_val))
    # print('The computed SNR value is ', dSNR)
    # print('Signal energy in this block : ', dSigEng)
    # print('Noise energy in this block : ', dNoiseEng)
    # print('Computed SNR : ', snr)
    return snr


def compute_wada_snr(path, time_SNR_window=30, min_time_SNR_window=3):
    """
    Calcul le SNR avec la méthode WADA d'un long (ou pas) fichier audio.
    Si le fichier est trop long retourne la moyenne des snrs des sous fenêtres de time_SNR_window secondes.
    La dernière fenêtre est ignoré si elle est inférieur à min_time_SNR_window secondes
    :param path: path of audio file
    :param time_SNR_window: temps en seconde des fenêtres utilisées (20secondes par défaut)
    :param min_time_SNR_window: temps en seconde minimal de la dernière fenêtres (2secondes par défaut)
    :return: le snr (float) calculé avec la méthode WADA pour le fichier audio.
    """
    if time_SNR_window < min_time_SNR_window:
        print("ABORT : ATTENTION time_SNR_window doit être bien plus grand que min_time_SNR_window")
    snrs = []
    duration = get_duration(path)
    if duration > time_SNR_window:
        start_list = numpy.arange(0, duration, time_SNR_window)
        for start in start_list:
            # print("{0}%".format(int(((start+time_SNR_window)/duration)*100)))
            # gestion de la dernière fenêtre temporelle
            if duration - start < time_SNR_window:
                time_SNR_window = duration - start
            # si la dernière fenêtre temporelle est insuffisante on va préférer l'ignorer
            if time_SNR_window > min_time_SNR_window:
                snrs.append(compute_snr(path, start=start, duration=time_SNR_window))
        return numpy.mean(numpy.array(snrs))
    else:

        return compute_snr(path, start=0, duration=int(duration))


def compute_wada_snr_seg(path, time_SNR_window=3, last_window=False):
    """
        Calcul le SNR segmentale avec la méthode WADA sur des fenêtre de time_SNR_window secondes.
        La dernière fenêtre est ignoré (car trop petite) si last_window est à False (par défault).
        :param path: path of audio file
        :param time_SNR_window: temps en seconde des fenêtres utilisées (3secondes par defaut)
        :param last_window: dernière fenêtre utilisé True, ou pas utilisée False (False par défaut)
        :return: le snr segmentale (list[float...]) calculé avec la méthode WADA sous forme de list.
        """
    snrs = []
    duration = get_duration(path)
    if duration > time_SNR_window:
        start_list = numpy.arange(0, duration, time_SNR_window)
        # on ignore la dernière fenêtre par défault car pas assez grande ( a commenter si vous voulez la conserver)
        if not last_window:
            start_list = start_list[:-1]
        for start in start_list:
            snrs.append(compute_snr(path, start=start, duration=time_SNR_window))
    else:
        snrs.append(compute_snr(path))
    return snrs
