# Multimodal Emotion Recognition 

Derek Stratton

Emily Hand

This repo contains a bunch of scripts for training emotion recognition models in the
text, audio, and visual modalities. The two datasets experimented on are [IEMOCAP](https://sail.usc.edu/iemocap/) and
[CMU MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/). IEMOCAP can be
requested for research use and CMU MOSI is publicly available, but they are needed to
run their respective experiments.

For reading in the datasets, I use the [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK)
and [torchemotion](https://github.com/alanwuha/torchemotion). I changed some parts of
torchemotion to suit my needs, which is why it's in this repo.
