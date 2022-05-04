## <b>DRPI-CGAN: Detection and Restoration of Photoshopped Images using Convolutional GAN </b>
[Mingxuan Teng](https://github.com/AlveinTeng),
[Jiakai Shi](https://github.com/VictorS67),
[Jiaming Yang](https://github.com/Jiaming-Yang-20)

This code provides the DRPI-CGAN model for solving problems of facial image manipulation detection and recovery.

![Semantic Diagram](https://drive.google.com/file/d/1WR_6rYMQx3ZYYdJvPk7yykPUMdthrI82/view?usp=sharing)

## Setup

### Install packages
- This code supports Python 3.7+.
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- The correlation layer in PWC-Net is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided [binary packages](https://docs.cupy.dev/en/stable/install.html#installing-cupy) as outlined in the CuPy repository.

### Download model weights
- Download pretrained weights files from [here](https://drive.google.com/file/d/1RyWliMj78-fALepCq7PBTBqB5Bf98TIh/view?usp=sharing).
- Upzip weights file. You should see `discriminator.pth` and `generator.pth`. These are pretrained weights files for discriminator and generator respectively.

### Download DRN-C-26 pretrained weights
- This code supports running local flow detection by using pretrained weights of DRN-C-26.
- Run `bash drn/weights/download_weights.sh` from the root directory of this repository.
- You should see `global.pth` and `local.pth`, where `local.pth` is the pretrained weights for local flow detection.

## How to run?

**Note:** Our models are trained on faces cropped by the dlib CNN face detector. Although in both scripts we included the `--no_crop` option to run the models without face crops, it is used for images with already cropped faces. Please make sure your images have the size of 400 * 400 if you choose no crop.

### Overfitting

To overfit the DRPI-CGAN model to a single pair of modified image and original image, run the following command from the root directory of this repository:

```bash
python main.py overfit --basepath=<path_to_your_dataset> --outputpath=<path_to_your_outputs>
```

This command will write loss values in `losses.csv` to `<your_path_to_outputs>`.

### Training

To train the DRPI-CGAN model on the training dataset, run the following command from the root directory of this repository:

```bash
python main.py train --basepath=<path_to_your_dataset> --outputpath=<path_to_your_outputs> --gen_checkpoint=<path_to_pretrained_generator_weights> --dis_checkpoint=<path_to_pretrained_discriminator_weights> 
```

This command will write loss values in `losses.csv`, generator/discriminator checkpoints and visualizations to `<your_path_to_outputs>`.

### Evaluation

To train the DRPI-CGAN model on the validation dataset, run the following command from the root directory of this repository:

```bash
python main.py evaluate --basepath=<path_to_your_dataset> --outputpath=<path_to_your_outputs>
```

This command will write 4 evaluation result csv files and visualizations to `<your_path_to_outputs>`.

### Local Flow Detection By Using Pretrained DRN-C-26

This code supports to run the local flow detection in [FALDetector](https://github.com/PeterWang512/FALdetector) for performance comparison with DRPI-CGAN.

To run the pretrained DRN-C-26 model on the validation dataset, run the following command from the root directory of this repository:

```bash
python main.py pretrain --basepath=<path_to_your_dataset> --outputpath=<path_to_your_outputs>
```

This command will write 4 evaluation result csv files and visualizations to `<your_path_to_outputs>`.

### Testing DRN-C-26 and PWC-Net

This code also supports to test the usability of DRN-C-26 and PWC-Net for generating optical flow.

To test the pretrained DRN-C-26 model and PWC-Net model on a single image pair, run the following command from the root directory of this repository:

```bash
python test.py --modify=<path_to_your_modified_image> --origin=<path_to_your_original_image> --model="./drn/weights/local.pth" --output_dir=<path_to_your_outputs>
```

## Dataset
A validation set consisting of 500 original and 500 modified images each from Flickr-Faces-HQ Dataset (FFHQ) on Kaggle can be downloaded [here](https://drive.google.com/file/d/1BJHppRV8Od6lYgEGbSX71h5gSd0rYxzK/view?usp=sharing).

After unzipping the file, you should see two folder `modified` and `reference`, where `reference` contains 500 original images and `modified` contains 500 modified images.

## Acknowledgments

This repository borrows partially from the [DCGAN Pytorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html), [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [FALDetector](https://github.com/PeterWang512/FALdetector), [drn](https://github.com/fyu/drn), [pwc-net](https://github.com/sniklaus/pytorch-pwc) and the PyTorch [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories. 
