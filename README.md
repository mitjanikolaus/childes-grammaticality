# CHILDES Grammaticality Annotations

Automatic annotation of grammaticality for child-caregiver conversations.

## Python environment

A python environment can be created using the [environment.yml](environments/environment.yml) file (for
GPU: [environment_gpu.yml](environments/environment_gpu.yml)):

```
conda env create --file environments/environment.yml
```

To install the current repo:
```
pip install .
```

Additionally, we need to install [my fork of the pylangacq repo](https://github.com/mitjanikolaus/pylangacq) (The original repo can be found here: [pylangacq](https://github.com/jacksonllee/pylangacq)) using pip:
```
git clone git@github.com:mitjanikolaus/pylangacq.git
cd pylangacq
source activate cf
pip install .
```

## Preprocess data


### CHILDES corpora
All English CHILDES corpora need to be downloaded from the
[CHILDES database](https://childes.talkbank.org/) and extracted to `~/data/CHILDES/`.

To preprocess the data, once you've installed the [pylangacq](https://github.com/mitjanikolaus/pylangacq) library as
mentioned above, you can run:

```
python preprocess.py
```
This preprocessed all corpora that are conversational (have child AND caregiver transcripts), and are in English.

Afterwards, the utterances need to be annotated with speech acts. Use the method `crf_annotate` from the following
repo: [childes-speech-acts](https://github.com/mitjanikolaus/childes-speech-acts).
```
python crf_annotate.py --model checkpoint_full_train --data ~/data/communicative_feedback/utterances_annotated.csv --out ~/data/communicative_feedback/utterances_with_speech_acts.csv --use-pos --use-bi-grams --use-repetitions
```

Finally, annotate speech-relatedness and intelligibility (this is used to filter out non-speech-like utterances and
non-intelligible utterances before annotating grammaticality):
```
python annotate_speech_related_and_intelligible.py
```

## Train models for annotation

### Baselines

Example for bigram-based model:
```
python grammaticality_annotation/train_grammaticality_baseline.py --model svc --max-n-gram-level 2
```

### Transformer-based models

These models are only fine-tuned on the task. Example for DeBERTa with context length 8:
```
python grammaticality_annotation/fine_tune_grammaticality_nn.py --model microsoft/deberta-v3-large --context-length 8
```


## Annotate data

```
python grammaticality_annotation/annotate_grammaticality_nn.py --model lightning_logs/version_1918412 --data-dir data/manual_annotation/all
```

The data will be annotated with the following coding scheme:

|  ungrammatical  | ambiguous | grammatical  |
|:---------------:|:---------:|:------------:|
|        0        |     1     |      2       |




## Acknowledgements
Thanks to the authors of the pylangacq repo: 

Lee, Jackson L., Ross Burkholder, Gallagher B. Flinn, and Emily R. Coppess. 2016.
[Working with CHAT transcripts in Python](https://jacksonllee.com/papers/lee-etal-2016-pylangacq.pdf).
Technical report [TR-2016-02](https://newtraell.cs.uchicago.edu/research/publications/techreports/TR-2016-02),
Department of Computer Science, University of Chicago.
