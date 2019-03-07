# EEG robot control

This project is a part of the  Biosignals Processing class at Technical University of Munich.
Goal: Development of EEG based BCI, used for control of a robot.


#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project is development of Brain-Computer Interface paradigm, capable of robot control with simple EEG-headset.

### Methods Used
* EEG
* Repetitive Visual Stimuli (RVS)
* Steady State Visually Evoked Potentials (SSVEP)
* Canonical Correlation Analysis (CCA)
* LDA and SVM

### Technologies
* Python
* [ratCAVE](https://github.com/ratcave)
* Pandas, jupyter, sklearn
* OpenVibe
* arduino


## Project Description

<figure>
<p align="center">
    <img src="/img/blockdiagram_analysis.png" width="80%">
    </p>
<figcaption>
    Project pipeline.   
</figcaption>
</figure><br>


In this project we focused on implementing a system that is capable to control an arduino based robot with EMOTIV EPOC+ EEG headset. The data was acquired with use of OpenVibe software and later on streamed to the Python script. The SSVEP experiment was designed in two ways, first one internally (used for final working implementation) and second one solely in ratCAVE, for possible migration to only Python based approach.

The second part of the pipeline was responsible for band-pass filtering, feature extraction and classification. For the final implementation I implemented CCA algorithm based on which SVM classification was applied.

### Other Team Members:
* [Yağmur Yener](https://github.com/reneyagmur)
* [Suleman Abbas](https://github.com/ssaz5)
* [Carlo DeDonno](https://github.com/HX4G0N)

### Acknowledgments:

* [Institute for Cognitive Systems, Technical University Munich](http://www.ics.ei.tum.de/en/home/) - for supervision and hardware access
