# Differentially Private Preference Data Synthesis

This repository contains the implementation of the paper "Differentially Private Preference Data Synthesis for Large Language Model Alignment".


## Requirements

```bash
conda create -n dpprefsyn_env python=3.9.21
conda activate dpprefsyn_env
pip install -r requirements.txt
```

## Experiments

The following example shows how to run DPPrefSyn on the OpenAssistant QA task at epsilon = 8:


```
python DPPrefSyn_OA.py --eps 8
```

Similarly, you can run DPPrefSyn on the Anthropic-HH QA task (`DPPrefSyn_HH.py`) and the TL;DR summarization task (`DPPrefSyn_summarize.py`).

## Questions

If anything is unclear, please contact Fengyu Gao (itsmefengyu@gmail.com).

