# Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach

We present models which complete missing text given transliterations of ancient Mesopotamian documents, originally written on cuneiform clay tablets (2500 BCE - 100 CE), to appear at EMNLP 2021 (Lazar et al., 2021).


## Setup

### Install requirements

Start by creating a virtual environment using the supplied requirements.txt file.

To scrape the Akkadian texts from ORACC:
```shell script
python preprocessing/scraping.py
```

### Getting the data

To scrape and preprocess the Akkadian texts and English translation of ORACC:
```shell script
python preprocessing/main_preprocess.py --do_scraping --do_preprocessing
```
We used the flags ``--remove_subscripts`` and ``--remove_superscripts`` in our models.
You can choose to exclude English translations with the flag ``--exclude_english``.
The english translations are created using multiple HTTP requests, hence this stage might take a while.

## Training

Training is done by Huggingface and has two modes: training from scratch a mini BERT model and fine-tuning a Multilingual BERT model.
You can switch between the two modes with the ``--from_scratch`` flag.
To train from scratch run:
```shell script
python akkadian_bert/main_bert.py --do_train --include_english --model_dir <model_dir> --from_scratch
```
To fine-tune with the masked language modeling task run:
```shell script
python akkadian_bert/main_bert.py --do_train --include_english --model_dir <model_dir>
```

You can choose to exclude English translations by removing the ``--include_english`` flag.
You can change the batch size and number of epochs by setting the ``batch_size`` and ``epochs`` flags, respectively.
The ``model_dir`` should specify a directory to save the trained model.

## Evaluation

Finally, to automatically evaluate the model you can run:
```
python main_bert.py --do_eval --model_dir <model_dir> --metrics_file <metrics_file>
```
When the ``metrics_file`` should be path to a pickle file to save the evaluation results, and the ``model_dir`` should be the directory of the trained model.

Any of the mentioned commands can be runned with a ``--help`` flag to get a full description of their different possible arguments.

## Citing 
```
@misc{lazar2021filling,
      title={Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach}, 
      author={Koren Lazar and Benny Saret and Asaf Yehudai and Wayne Horowitz and Nathan Wasserman and Gabriel Stanovsky},
      year={2021},
      eprint={2109.04513},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
