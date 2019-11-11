# Proj1

> Write your own code to realize two-dimension convolution and some image filters.
> Requirement：
> * Program to realize the convolution operation and the next filters
> * Roberts operator; Prewitt operator; Sobel operator;
> * Gaussian filter, mean filter and Median filter
> * Kernal size and sigma adjustable
> * Design proper UI and display I/O images
> 
> language：python or C++

## Usage

### Dependencies

This project is implemented in Python. You'll need `PyQt5` and `opencv-python` to run this.

```bash
pip3 install opencv-python PyQt5
```

If you want to take a look at the notebook, you'll need `jupyter notebook` and `matplotlib` as well. Install them by yourself.

Explanation for each file:

* `main.py`: GUI implmentation;
* `ops.py`: Convolution & Image filters implementations;
* `presentation.ipynb`: Jupyter notebook to give an intuitive presentation of effects.

To run the GUI, just run:

```bash
python3 main.py
```

This will open `lenna.png` in grey scale mode and you can operate on that image. To use another file:

```bash
python3 main.py <filename>
```