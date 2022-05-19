
# L2-NLP—Training POS + Dependency parser with spaCy

This repo contains necessary codes to train a series of spaCy models in Kyle et al. (2022).
The current repo uses [spacy ud_benchmark project](https://github.com/explosion/projects/tree/v3/benchmarks/ud_benchmark) as a template.


## TO DO list
- [ ] Add more description on this readme.
- [ ] 

## Prerequisite
- Python
- spaCy version 3

## How to get the corpus
- TBA

## Steps to reproduce the result
The following steps should be done to reproduce our result on the paper.
### Put the corpus file under Asset folder

### Edit project.yml file

### Running preprocessing script

For each of the dataset to train, we should run the following command.

```
python -m spacy project run convert
```

This command runs a script which takes the `.conll` files and creates `.spacy` object under `corpus` directory. This is done for the specific data specified in the `project.yml` file. More specifically, `ud_prefix` variables specifies which dataset to convert.


### Run command line
```
python -m spacy project run all
```