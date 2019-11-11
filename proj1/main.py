#!/usr/bin/env python3
import sys
from collections.abc import Iterable, Generator
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from ops import *

# load lenna
img_name = "lenna.png" if len(sys.argv) == 1 else sys.argv[1]
img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

def getQPixmapFromNpArray(arr):
    arr = arr.astype("uint8")
    if arr.ndim == 2:
        height, width = arr.shape
        return QPixmap.fromImage(QImage(arr.data, width, height, width, QImage.Format_Grayscale8))
    else:
        height, width, channel = arr.shape
        bytesPerLine = channel * width
        return QPixmap.fromImage(QImage(arr.data, width, height, bytesPerLine, QImage.Format_RGB888))

class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        # set layout
        self.initLayout()
        # set initial states
        self.kernel_func = None
        self.kernel_size = 3
        self.sigma = None
    
    def initLayout(self):
        self.img_label = QtWidgets.QLabel()
        self.img_label.setPixmap(getQPixmapFromNpArray(img))

        kernel_settings_layout = QtWidgets.QFormLayout()
        size_settings_layout = QtWidgets.QFormLayout()
        
        self.edge_kernels_selections = self.getGroupbox(
            "Edge detection kernels", self.onChangeEdgeDetectionKernel, edge_kernels)
        self.mean_kernels_selections = self.getGroupbox(
            "Mean kernels", self.onChangeBlurKernel, mean_kernels)
        self.kernel_size_selections = self.getGroupbox(
            "Set kernel size", self.onChangeKernelSize, range(3, 19, 2))
        self.sigma_selections = self.getGroupbox(
            "Set sigma", self.onChangeSigma, [i/2 for i in range(2, 11)])
        kernel_settings_layout.addWidget(self.edge_kernels_selections)
        kernel_settings_layout.addWidget(self.mean_kernels_selections)
        size_settings_layout.addWidget(self.kernel_size_selections)
        size_settings_layout.addWidget(self.sigma_selections)
        self.kernel_size_selections.setHidden(True)
        self.sigma_selections.setHidden(True)
        kernel_settings_form = QtWidgets.QWidget()
        kernel_settings_form.setLayout(kernel_settings_layout)
        size_settings_form = QtWidgets.QWidget()
        size_settings_form.setLayout(size_settings_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.img_label)
        layout.addWidget(kernel_settings_form)
        layout.addWidget(size_settings_form)
        self.setLayout(layout)
    
    @staticmethod
    def getGroupbox(title, handler, it, attr_name = "value"):
        ret = QtWidgets.QGroupBox()
        ret.setTitle(title)
        
        layout = QtWidgets.QFormLayout()
        if type(it) == dict:
            ite = it.items()
        elif isinstance(it, Iterable):
            if isinstance(it, Generator):
                print("Warning: using generator as iterable.")
            ite = zip(map(str, it), it)
        
        for k,v in ite:
            a = QtWidgets.QRadioButton(k)
            setattr(a, attr_name, v)
            a.toggled.connect(handler)
            layout.addWidget(a)
        
        ret.setLayout(layout)
        return ret
    
    def onChangeBlurKernel(self, toggled):
        if not toggled:
            return
        sender = self.sender()
        self.kernel_func = sender.value
        self.kernel_size_selections.setHidden(False)
        if self.kernel_func == gaussian:
            self.sigma_selections.setHidden(False)
            self.refreshImage(True, True)
        else:
            self.sigma_selections.setHidden(True)
            self.refreshImage(True, False)

    def onChangeKernelSize(self, toggled):
        if not toggled:
            return
        sender = self.sender()
        self.kernel_size = sender.value
        self.refreshImage(True)

    def onChangeEdgeDetectionKernel(self, toggled):
        if not toggled:
            return
        sender = self.sender()
        self.kernel_func = sender.value
        self.kernel_size_selections.setHidden(True)
        self.sigma_selections.setHidden(True)
        self.refreshImage(False)
    
    def onChangeSigma(self, toggled):
        if not toggled:
            return
        if self.kernel_func != gaussian:
            # not gaussian... prompt user
            self.sigma_selections.setStyleSheet("color: red")
        else:
            self.sigma_selections.setStyleSheet("")
            self.sigma = self.sender().value
        
        self.refreshImage(True, True)

    def refreshImage(self, use_kernel_size = False, use_sigma = False):
        if use_kernel_size:
            if use_sigma:
                pm = getQPixmapFromNpArray(self.kernel_func(img, kernel_size = self.kernel_size, sigma = self.sigma))
            else:
                pm = getQPixmapFromNpArray(self.kernel_func(img, kernel_size = self.kernel_size))
        else:
            pm = getQPixmapFromNpArray(self.kernel_func(img))
        self.img_label.setPixmap(pm)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(512, 512)
    w.move(300, 300)
    w.setWindowTitle('Proj1 convolution by Eugene')
    w.show()
    sys.exit(app.exec_())
