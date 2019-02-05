# End-to-End Recognition of Handwritten Notes in Scanned Documents

This repository contains all files necessary to train, run, and evaluate the sytem presented in the Master's Thesis "End-to-End Recognition of Handwritten Notes in Scanned Documents".
With the end-to-end system, you can transcribe (only the) handwritten notes on pages, even when they are mixing with printed text.
This repository does not include any models; however you can download some here or train them yourself.

A good dataset to train the model on is the [Paper Notes](https://www.github.com/torbenm/paper-notes) dataset.

In the following, the Training, Running, Evaluation (with Ceiling Analysis) is described.

## Setup

The system was implemented using Tensorflow 1.9, Python 3.7, CUDA 10, and cuDNN 7.4.
All networks were evaluated on a system with a GeForce GTX 1080 Ti GPU with about 11 GB of memory, an Intel i9-7900X CPU with 3.3 GHz, and about 64 GB of RAM.

To install the dependencies, run `pip install -r requirements.txt`.
There might be some missing even though I tried to keep the file complete.

## Training

The system is made up of several building blocks, which can be combined together:

<img src="docs/pipeline.png" width="400" />

In this section, training the deep learning components (i.e. Text Separation, Text Classification, Text Recognition) as well as setting up the Language Model is described.

### Gathering the data

First you should download the IAM dataset. For this, run

```
python -m data.download --dataset iam --user USERNAME --pwd PASSWORD
```

`PASSWORD` and `USERNAME` should be replaced with the respective values you set up to access the [IAM database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).

Next, you need to prepare the datasets for training.
This can be done with

```
python -m data.prepare --config iam-both-small
```

The config parameter tells the program what configuration to consult. There are two configurations provided: `iam-both-small` and `iam-both-large`. You can create your own configurations by placing a `.json` file in the `/data/config/` folder.

If you want to base the splitting into train, test, dev on the Paper Notes dataset, you should run it with the following command.
Here, you of course should adjust the path yourself to the according folder.

```
python -m data.prepare --config iam-both-small --split ../paper-notes/data/final
```

The following options are available:

```jsonc
{
  "dataset": "iam", // The base dataset
  "subset": "both", // Which subset to use. "both" describes words and lines for iam
  "limit": -1, // Limit of samples (-1 is infinite)
  "shuffle": true, // Whether or not to shuffle
  "test_frac": 0.1, // Fraction for testing
  "dev_frac": 0.1, // Fraction for development
  "invert": true, // Whether to invert the files
  "threshold": false, // Whether to apply thresholding
  "crop": true, // Whether to crop the image to the smallest part of non-background pixels
  "scale": {
    "mode": "none" // How to scale the image. Modes are "factor",
    //"none", "height", "line" (see scale.py for more detail)
  },
  "padding": 5, // Number of pixels to padd with
  "binarize": false, // Whether to binarize
  "printed": {
    // If this option is included you add printed text versions
    "minlength": 2, // Minimum length of text to make it printed
    "count": [10000, 1000, 1000], // Number of samples per [train, dev, test]
    "padding": 5, // Padding to apply
    "printing_padding": 5, // padding of the printed text
    "crop": true, // Whether to crop the printed text
    "height": {
      "min": 10, // Minimum height
      "max": 110, // Maximum heihgt
      "center": [20, 40, 60, 80, 100], // Centers of gaussian distributions for random height selection
      "scale": 5
    },
    "foreground": {
      "low": 0, // The values the foreground pixels can be betrween
      "high": 0
    },
    "filters": [],
    "fonts": [
      // Which fonts to use
      "./data/fonts/AppleGaramond.ttf",
      "./data/fonts/Cursif.ttf",
      "./data/fonts/EnglishTowne.ttf",
      "./data/fonts/LinLibertine.ttf",
      "./data/fonts/OldNewspaperTypes.ttf",
      "./data/fonts/Roboto-Light.ttf",
      "./data/fonts/Asar-Regular.ttf",
      "./data/fonts/Bellefair-Regular.ttf",
      "./data/fonts/JosefinSlab-Thin.ttf",
      "./data/fonts/JosefinSlab-Light.ttf",
      "./data/fonts/Lato-LightItalic.ttf",
      "./data/fonts/Lato-Hairline.ttf",
      "./data/fonts/LibreBaskerville-Italic.ttf",
      "./data/fonts/LibreBaskerville-Regular.ttf",
      "./data/fonts/PlayfairDisplay-BoldItalic.ttf",
      "./data/fonts/PlayfairDisplay-Regular.ttf",
      "./data/fonts/Tangerine-Regular.ttf",
      "./data/fonts/TulpenOne-Regular.ttf",
      "./data/fonts/Ubuntu-MediumItalic.ttf",
      "./data/fonts/IMFeENsc28P.ttf"
    ]
  }
}
```

### Text Separation

To train the `Text Separation Network`, you first need to define a configuration file. In my thesis, I used `config/separation/base.json`.
Here are some of the options.

```jsonc
{
  "name": "separation", // name of the configuration
  "save": 2, // after which epoch to make a checkpoint (here, every second epoch)
  "binary": true, // Whether to binarize the data
  "max_batches": {
    "sep": {
      // MAximum number of batches per subset
      "train": 1000,
      "val": 1000 // same as "dev"
    }
  },
  "data_config": {
    "paper_notes_path": "../paper-notes/data/final", // Path to the paper notes dataset
    // How the page should be separated into tiles/slices
    "slice_width": 512,
    "slice_height": 512,
    "filter": true, // Whether to filter out slices without handwriting
    "otf_augmentations": {} // see data/ImageAugmenter.py for options
    // empty object means no augmentations are applied
    // otf stands for "on the fly", thus there will be no two batches alike
  },
  "batch": 4, // Size of a batch
  "learning_rate": 0.0000001, // Size of the learning rate
  "algo_config": {
    "filter_size": 5, // Filter size of the convolutions
    "features_root": 32, // Number of features in the first convolution (doubled every contracting block)
    "batch_norm": false, // whether to use batch norm
    "group_norm": 32 // whether to use group norm and the group size
    // ...  more options can be found in ./nn/tfunet.py
  }
}
```

To start the actual training, execute

```sh
python train_separation.py --config separation/base [--gpu GPU] [--model-date] [--model-epoch]
```

You should pass a GPU parameter with the index of the GPU to use. Default is `-1`, thus CPU execution.
The model snapshots are stored in `models/$NAME_OF_THE_CONFIG_$CURRENT_DATE/`. Therefore, if you want to continue training an existing model, you should pass `--model-date` set to the according date and `--model-epoch` to the epoch where you want to continue.
Of course, you can replace `separation/base` with whatever configuration you want to use (in the same schema, without the `config` folder name and without `.json`).

### Text Classification & Recognition

Training the text classification and recognition network are very similar to the text separation network.
Since they are interleaved as multi-task learning networks,
they are both trained at once.
The configuration used in my thesis is `config/otf-iam-paper-fair.json`.

The following options in the configuration are available:

```jsonc
{
  "name": "otf-iam-paper", // name of the configuration
  "dataset": "iam-both-small", // which dataset to use (go by filename)
  "algorithm": "htrnet", // You should stick to this
  "save": 3, // After which epoch to create a snapshot of the model
  "batch": 50, // Batch size
  "epochs": 1000, // number of epochs
  "max_batches": {
    "rec": {
      "train": 400, // numberof batches for recognition training
      "dev": 200 // number of batches for recognition validation
    },
    "class": {
      "train": 300, // number of batches for classification training
      "dev": 100 // number of batches for classificaiton development
    }
  },
  "learning_rate": 0.0003, // Learning reate for recognition
  "class_learning_rate": 0.0001, // Learning rate for classification
  "ctc": "greedy", // Which decoder to use for ctc
  "data_config": {
    "shuffle_epoch": true,
    "type_probs": {
      // Probability to select a given subset
      // This should be chosen according to the amount of words/lines
      // In this case, it yields roughly 50-50 in the training
      "lines": 1,
      "words": 0.12
    },
    "otf_augmentations": {
      // similar to text separation
      "affine": {}, // empty object here means that the standard settings from data/steps/pipes/affine.py
      "warp": {
        "prob": 0.5,
        "deviation": 1.35,
        "gridsize": [15, 15]
      }
    },
    "preprocess": {
      // Which preprocessing steps to apply with
      "crop": true,
      "invert": true,
      "scale": 2.15,
      "padding": 5
    },
    "postprocess": {
      // This is not applied after the training, but after preprocessing & augmentation
      "binarize": true
    }
  },
  "algo_config": {
    // Settings for the algorithm itself
    "format": "nchw",
    "scale": false,
    "encoder": {
      "type": "cnn",
      "cnn": {
        "bias": true,
        "dropout": {
          "active": true
        },
        "bnorm": {
          "active": true,
          "train": true
        }
      }
    },
    "classifier": {
      "units": [512] // which dense layers are added
    },
    "fc": {
      "use_activation": false
    },
    "optimizer": "adam"
  }
}
```

To start the actual training, execute

```sh
python train.py --config otf-iam-paper-fair [--gpu GPU] [--model-date] [--model-epoch] [--no-trains] [--no-class] [--paper-notes] [--only-paper]
```

The config, gpu, model date and model epoch are similar to the text separation network. With `--no-trans` and `--no-class` you can skip training on either the recognizer or the classification part. With `--paper-notes`, you can add training on the extracted words of the Paper Notes dataset.
With `--only-paper`, the models are _only_ trained on the Paper Notes dataset.
With the Paper Note options, you should also set `--paper-note-path` to the acording path if it deviates from `../paper-notes/data/words`

### Language Model

To create the language model, you need to download and place the brown / lob corpus in the folder `data/raw/corpus`.
If you want to train on other files, you should check out `build_corpus.py` to see what's necessary (also if you want to adjust the path to the paper notes, which causes some words to be removed that are appearing in the evaluation/testing data).

Run the following to create the corpus then:

```
python build_corpus.py
```

The corpus is then created as `./data/output/corpus.json`.
However, the corpus used in my thesis is also commited as `data/corpus.json` (this is the location where the Language Model will look for the corpus, so either replace that file or change the language model settings).

### Word Detector, Paragraph Detector, Line Segmentation & Co

These models are currently trained by simply evaluation every parameter (see in the evaluation section below).

## Running

### Single Building Blocks

### A whole pipeline

## Evaluating

## Ceiling Analysis
