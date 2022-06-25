
# L2-NLP—Training POS + Dependency parser with spaCy

This repo contains necessary codes to train a series of spaCy models in Kyle et al. (2022).
The current repo uses [spacy ud_benchmark project](https://github.com/explosion/projects/tree/v3/benchmarks/ud_benchmark) as a template.


## Prerequisite
- Python
- spaCy version 3

## Getting the training corpus
- TBA 

# Steps to reproduce the result
Please follow the following steps to reproduce our result on the paper.

## Put the corpus file under `assets` folder
Once you obtained the training corpus files in  `.connlu` format, put them under `assets/ud` and `assets/pos` respectively. Spacy project commands will recognize these `conllu` files when prompted and convert them to trainable spacy `doc` object for training (see more details below).

## Edit project.yml file
The `project.yml` file provides all the command necessary to access the training configulation and the corpus file. To train each of the `ud` and `pos` pipeline with different configulation (either `trf` or `t2v`), you can change the variables in this `project.yml` file. The following is the example of the variable section to train `ud` with `trf` pipeline with `L1L2e_combined` dataset.

```yml
# Variables can be referenced across the project.yml using ${vars.var_name}
vars:
  config: "trf" # Choose from "t2v" or "trf"; set config file each time you train model
  ud_treebank: "UD_English-EWT" #don't change this
  ud_prefix: "L1L2e_combined" #training dataset: choose from L1, L1L2_combined, L1L2e_combined
  pipeline: "ud" # use ud for POS + DEP; pos for POS only.
  spacy_lang: "en" #don't change this
  # spacy model with vectors by name or path: "en_core_web_lg"
  # if used, switch from train to train-with-vectors
  vectors: null 
  gpu: -1 #we don't use GPU, if you have GPU device with cuda, set 0
  package_version: "0.0.1" #this will be used as the final package version
```


## Run preprocessing script

The following command converts the raw `conllu` file into spacy `doc` object for training.

```bash
python -m spacy project run convert
```

This command runs a script which takes the `.conll` files and creates `.spacy` object under `corpus` directory. It will automatically create the directories if not present.

The script only converts the dataset which is specified in the `project.yml` file (see above).
For instance, if you set `ud`, `L1L2e_combined` in the `yml` file, the script converts the following three corpus files.
- L1L2e_combined-ud-dev.conllu
- L1L2e_combined-ud-test.conllu
- L1L2e_combined-ud-train.conllu

To convert other files, change the settings in `project.yml` file and run the command again.


## Run training command

When you successfully converted the `.conllu` in the `assets` folder into `.spacy` in the `corpus` folder, you can run the following command to start training. Note that spacy uses `training` and `dev` data in this step.

```bash
python -m spacy project run all
```

This will create a specific directory under `training` folder. When the training step is completed, it will also create spacy package under the `package` folder. 

In the paper, we did not use `evaluate` command from spacy, but we create a separate script to evaluate the parsers accuracy using our own evaluation script.


## TO DO list
- [x] Add more description on this readme.
- [ ] Check reproducibility
- [ ] Add more description on evaluation script 