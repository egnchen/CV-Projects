# Proj2

> Author: 陈奕君
>
> Student No: 517021910387

> Write your own code to realize these functions:

> * Morphological edge detection
> * Morphological Reconstruction
>   * Conditional dilation in binary image
>   * Gray scale Reconstruction
>   * Morphological gradient
> 
> Requirements:
> 
> Design your own UI and display I/O images
> User customized SE
> Language：python or C++


## Usage

### Dependencies

This project is implemented in Python. You'll need `PyQt5` and `opencv-python` to run this.

```bash
pip3 install opencv-python PyQt5
```

In my proj2 code, I used the labelling algorithm implemented in `scikit-image` package, so **you'll have to install that too**.

```bash
pip3 install scikit-image
```

If you want to take a look at the notebook, you'll need `jupyter notebook` and `matplotlib` as well. Install them by yourself.

### Running the program

Explanation for each file:

* `main.py`: GUI implmentation;
* `ops.py`: Morphological operation implementations;
* `presentation.ipynb`: Jupyter notebook to give an intuitive presentation of effects.

To run the GUI, just run:

```bash
python3 main.py
```

This will open `lenna.png` in grey scale mode and you can operate on that image. To use another file:

```bash
python3 main.py <filename>
```

Due to performance issues, all original images are downscaled to smaller than `128x128` if they're larger than that. The images are processed under `128x128` but they're upscaled to at least `256x256` when displayed onto the screen, so that you can have a clearer view.

## Explanation

Most of the GUI are self-explanatory. There're a few things I would like to point out about the reconstruction operations:

`binary conditional dilation` receive a labelled image as marker, the original image as the mask and a SE as input. For labelled marker, the program use that computed from `bFinary get labelled marker image` **directly** and cannot be specified to external file. 

If you're receiving underwhelming results, try to tweak the iteration count of dilation in the dilation / labelling process. View the labelled image in `binary get labeled marker image` directly and tweak the settings until you get the desired result, than switch back to `binary conditional dilation`.

The same goes for `gray image reconstruction`. You can tweak the iteration count to specify the marker and see those results.