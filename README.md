# Distributionally Robust Policy Evaluation under General Covariate Shift in Contextual-Bandits

This is the repo for paper Distributionally-Robust-Policy-Evaluation-under-General-Co--variate-Shift-in-Contextual-Bandits. 


### Experiments
#### Policy shift:
```console
# logging policy known:
$ python PS_known_dm.py # direct method
$ python PS_known_robust.py # proposed methods

# logging policy unknown:
$ python PS_unknown_dm.py # direct method
$ python PS_unknown_robust.py # proposed methods
```
#### General covariate shift:
```console
# logging policy and covariate shift known:
$ python GCSS_known_dm.py # direct method
$ python GCS_known_robust.py # proposed methods

# logging policy and covariate shift unknown:
$ python GCS_unknown_dm.py # direct method
$ python GCS_unknown_robust.py # proposed methods
```
