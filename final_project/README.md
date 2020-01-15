# Final project: Bottle cap detection

Team work with Rui'en Shen, Yuxuan Liu & Yajie Yu.

## Requirements

> Task: Detect bottle caps in a picture. Ten bottle caps are presented in a picture with a certain background(pure color or some with complex texture) and your program should be able to detect all the caps along with their orientations(head, tail or side).
> Basic requirements: 10 bottle caps are presented with different orientations, and pure color is provided as background. Detect all of them correctly. Your program should be wrapped in a GUI.
> Bonus requirements: Your program should be able to detect bottle caps of different colors(10'), or different sizes & shapes(10'). Your program should be able to filter out various obstacles and other objects in the image(10'), or keep working with complex texture as background or tilted camera angles(20').

## Our solution

Our solution mainly consists of two parts: image segmentation and orientation detection. Segmentation is mainly done using morphological operations and watershed method. It can correctly detect bottle caps and separate them when they're adjacent to each other. Orientation detection is done by combining edge detection and SURF matching.

Presentations can be found in the jupyter notebook file. GUI can be found in `gui` folder. To run the notebook, you'll need following dependencies apart from `jupyter` itself:
```bash
sudo pip3 install opencv-python matplotlib
```

To run the GUI you'll also need `PyQt5` and some utility packages:
```bash
sudo pip3 install PyQt5 qimage2ndarray
```

The part each of us was responsible for:

* Image segmentation: courtesy of Yijun Chen(me).
* Cap orientation detection: courtesy of Yijun Chen(edge method) and Rui'en Shen(SURF method).
* GUI: courtesy of Yajie Yu.
* Report & presentation: courtesy of Yuxuan Liu.
