import sys
from collections.abc import Iterable, Generator
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from ops import *
import numpy as np
# some code are from proj1

# load lenna
img_name = "lenna.png" if len(sys.argv) == 1 else sys.argv[1]
img = load_image(img_name)

def getQPixmapFromNpArray(arr):
    cv2.imwrite("lastsaved.png", arr)
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
        self.op_title = "original image"
        self.op = operations[self.op_title]
        self.kernel_func = None
        self.kernel_size = 3
        self.sigma = None
        self.gray_se = self.binary_se = None
        self.ite_count = 10
        self.refreshImage()
        self.updateBinarySELabel()
        self.updateGraySELabel()
        self.updateIteCountLabel()
    
    def initLayout(self):
        self.img_label = QtWidgets.QLabel()
        # add button to change SE
        self.specify_binary_se_button = QtWidgets.QPushButton("Change binary SE")
        self.specify_binary_se_button.clicked.connect(self.onSpecifyBinarySEClicked)
        self.specify_gray_se_button = QtWidgets.QPushButton("Change gray SE")
        self.specify_gray_se_button.clicked.connect(self.onSpecifyGraySEClicked)
        # add button to change marker iteration count
        self.specify_ite_count_button = QtWidgets.QPushButton("Change Iteration count")
        self.specify_ite_count_button.clicked.connect(self.onSpecifyIteCountClicked)
        # current se label
        self.binary_se_label = QtWidgets.QLabel()
        self.binary_se_label.setWordWrap(True)
        self.gray_se_label = QtWidgets.QLabel()
        self.gray_se_label.setWordWrap(True)
        # current iteration count label
        self.ite_count_label = QtWidgets.QLabel()
        
        self.operation_selections = self.getGroupbox(
            "Operation selection", self.onChangeOperation, operations
        )
        operation_selection_layout = QtWidgets.QFormLayout()
        operation_selection_form = QtWidgets.QWidget()
        operation_selection_layout.addWidget(self.operation_selections)
        operation_selection_form.setLayout(operation_selection_layout)

        image_layout = QtWidgets.QVBoxLayout()
        image_layout.addWidget(self.img_label)
        image_layout.addWidget(self.binary_se_label)
        image_layout.addWidget(self.gray_se_label)
        image_layout.addWidget(self.ite_count_label)
        image_layout.addWidget(self.specify_binary_se_button)
        image_layout.addWidget(self.specify_gray_se_button)
        image_layout.addWidget(self.specify_ite_count_button)
        first_row = QtWidgets.QWidget()
        first_row.setLayout(image_layout)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(first_row)
        layout.addWidget(operation_selection_form)
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
    
    def onChangeOperation(self, toggled):
        if not toggled:
            return
        sender = self.sender()
        self.op_title = sender.text()
        self.op = sender.value
        self.refreshImage()
    
    def onSpecifyIteCountClicked(self):
        prompt="""
        Enter desired iteration count:
        This iteration count is account for:
        Time of binary erosion in marker image generation
        TIme of gray erosion in gray marker image generation
        """
        user_input, ok = QtWidgets.QInputDialog.getText(self, "Enter iteration count", prompt)
        if ok:
            ite = eval(user_input)
            try:
                ite = int(user_input)
            except ValueError:
                QtWidgets.QMessageBox.information(self, "Invalid input", "Only takes integer input.")
                return
            self.ite_count = ite
        else:
            QtWidgets.QMessageBox.information(self, "Info", "Iteration count not specified, will fallback to default.")
            self.ite_count = 10
        self.updateIteCountLabel()
        self.refreshImage()
    
    def onSpecifyGraySEClicked(self):
        prompt = """
        Enter your se in python list or python tuple
        for example: [[-1,-2,-3],[4,5,6],[-7,-8,-9]]
        You may enter negative values here.
        """
        invalid_input_prompt = """
        Oh, what you just entered is neither a list nor a tuple.
        Did you make a mistake, or you're just being a bad boy?
        """
        user_input, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Enter custom SE", prompt)
        if ok:
            se = eval(user_input)
            if type(se) not in (list, tuple):
                QtWidgets.QMessageBox.information(self, "Invalid input", invalid_input_prompt)
                return
            else:
                se = np.array(se)
                if se.ndim != 2 or se.shape[0] != se.shape[1]:
                    QtWidgets.QMessageBox.information(self, "Invalid input", "Input have invalid dimensions.")
                    return
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Successfully set", f"Gray SE set as:\n{se}\nwith a shape of {se.shape}")
                    self.gray_se = se.astype('int8')
                    self.refreshImage()
        else:
            QtWidgets.QMessageBox.information(self, "Info", "Gray SE not specified. Will fallback to none and automatically specified to default during next operation.")
            self.se = None

    def onSpecifyBinarySEClicked(self):
        prompt = """
        Enter your se in python list or python tuple
        for example: [[0,1,0],[1,1,1],[0,1,0]]
        You only can enter 0s and 1s in your se.
        """
        invalid_input_prompt = """
        Oh, what you just entered is neither a list nor a tuple.
        Did you make a mistake, or you're just being a bad boy?
        """
        user_input, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Enter custom SE", prompt)
        if ok:
            se = eval(user_input)
            if type(se) not in (list, tuple):
                QtWidgets.QMessageBox.information(self, "Invalid input", invalid_input_prompt)
                return
            else:
                se = np.array(se)
                if se.ndim != 2 or se.shape[0] != se.shape[1]:
                    QtWidgets.QMessageBox.information(self, "Invalid input", "Input have invalid dimensions.")
                    return
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Successfully set", f"Binary SE set as:\n{se}\nwith a shape of {se.shape}")
                    self.binary_se = se.astype('uint8')
                    self.refreshImage()
        else:
            QtWidgets.QMessageBox.information(self, "Info", "Binary SE not specified. Will fallback to none and automatically specified to default during next operation.")
            self.se = None

    def updateBinarySELabel(self):
        se_text = "None" if self.binary_se is None else str(self.binary_se)
        self.binary_se_label.setText("Current binary SE:\n" + se_text)
    
    def updateGraySELabel(self):
        se_text = "None" if self.gray_se is None else str(self.gray_se)
        self.gray_se_label.setText("Current gray SE:\n" + se_text)
    
    def updateIteCountLabel(self):
        self.ite_count_label.setText("Current iteration count(for labelling): " + str(self.ite_count))

    def refreshImage(self, use_kernel_size = False, use_sigma = False):
        if "binary" in self.op_title:
            se = binary_kernel if self.binary_se is None else self.binary_se
            self.binary_se = se
            self.updateBinarySELabel()
        elif "gray" in self.op_title:
            se = gray_kernel if self.gray_se is None else self.gray_se
            self.gray_se = se
            self.updateGraySELabel()
        else:
            print("Invalid title, cannot select se!")
            se = binary_kernel

        if self.op_title:
            if self.op_title.startswith("binary"):
                if "labelled" in self.op_title or "conditional dilation" in self.op_title:
                    im_bw = get_image_bw(img)
                    if "labelled" in self.op_title:
                        # gettin labels, so some specified code
                        current_img = get_marker(im_bw, ite_count=self.ite_count)
                    else:
                        # getting condition dilation
                        current_img = conditional_dilation(get_marker(im_bw, ite_count=self.ite_count), im_bw, kernel=se)
                    # remap the colors
                    max_label = np.max(current_img)
                    scale = 255 // max_label
                    current_img = scale * current_img
                else:
                    # normal binary ops
                    current_img = self.op(get_image_bw(img))
            elif self.op_title.startswith("gray"):
                if "reconstruction" in self.op_title:
                    current_img = self.op(img, kernel=se, ite_count=self.ite_count)
                else:
                    current_img = self.op(img, kernel=se)
            else:
                # get binary image or original image
                current_img = self.op(img)
        else:
            current_img = img
        pixmap = getQPixmapFromNpArray(current_img)
        pixmap = pixmap.scaledToWidth(256).scaledToHeight(256)
        self.img_label.setPixmap(pixmap)
        self.updateBinarySELabel()
        self.updateGraySELabel()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(512, 512)
    w.move(300, 300)
    w.setWindowTitle('Proj2 morphological operations by Eugene')
    w.show()
    sys.exit(app.exec_())
