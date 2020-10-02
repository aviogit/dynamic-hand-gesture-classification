# Dynamic Hand Gestures Classification for the SFINGE 3D dataset

Anonymous repository to host code and data to run the 3D hand gestures classification pipeline, based on Vispy and a ResNet-50 trained with Fast.ai, on the SFINGE 3D dataset.


<img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/one.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/two.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/three.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/four.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/expand.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/ok.png" width="150">

<img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/grab.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/pinch.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/swipe-left.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/swipe-right.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/swipe-V.png" width="150"><img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/swipe-O.png" width="150">

[//]: # (<img src="https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/tap.png" width="150">)
[//]: # (ffmpeg -ss 00:00:00.000 -i desktop-capture-20200512-191206.mp4 -pix_fmt rgb24 -r 10 -s 900x500 -t 00:00:24.000 output.gif)

## How to perform online inference on the SFINGE 3D dataset

- tested on Ubuntu 18.04 with CUDA already installed - inference performed on CPU (slower, but the setup is easier)

### Step 1: put everything in place (in `/tmp`, optimal choice for a disposable installation)

`#> cd /tmp`

Clone the SFINGE 3D dataset

`#> git clone git@github.com:SFINGE3D/DatasetV1.git`

Create a new virtualenv for this repo

`#> mkdir /tmp/dynamic-hand-gestures-venv`

`#> python3 -m venv /tmp/dynamic-hand-gestures-venv`

Activate the virtualenv

`#> source /tmp/dynamic-hand-gestures-venv/bin/activate`

Clone this repo

`#> git clone git@github.com:dynamic-hand-gestures-classification/dynamic-hand-gestures-classification.git`

Upgrade pip

`#> pip install --upgrade pip`

`#> cd /tmp/dynamic-hand-gestures-classification/`

Install the requirements

`#> pip install -r requirements.txt`


### Step 2: translate data files from the SFINGE 3D format to our format

`#> cd /tmp/dynamic-hand-gestures-classification/utilities/`

`#> ./conversion.py --filename /tmp/DatasetV1/Sequences/3.txt --csv-separator=';',`


### Step 3: perform online inference on the datafiles

If you get CUDA errors here, such as:

`ImportError: libcudart.so.9.0: cannot open shared object file: No such file or directory`

it could mean that your CUDA drivers are too old for Pytorch 1.4.0, so downgrade it with:

`#> pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html`

Unset `LD_LIBRARY_PATH` environment variable, just in case...

`#> cd /tmp/dynamic-hand-gestures-classification`

`#> unset LD_LIBRARY_PATH`

`#> ./dynamic-hand-gestures.py ./utilities/unknown-3.csv.xz --dataset-path ./utilities/ --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-04-21_15.47.18-SFINGE3D-dataset-transfer-learning-from-our-dataset-data-augmentation-with-partial-gestures-and-noise.pkl --cuda-device cpu --inference-every-n-frames 20 --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --fps 10 --save-image-only-when-prob-greater-than 0.98`

The VisPy visualizer should now go full screen and show the sequence as it is drawn and sent to the inference module (the ResNet-50 trained with Fast.ai).

![online inference demo](https://raw.githubusercontent.com/dynamic-hand-gestures-classification/dynamic-hand-gestures-classification/master/pics/sfinge3D-dataset/desktop-captures/desktop-capture-20200512-191206.gif "desktop-capture-20200512-191206")

##### For more information, please refer to the [usage page](usage.md)
