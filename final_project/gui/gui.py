from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.pyplot import imread, imsave
from cv2 import imwrite
import qimage2ndarray
import numpy as np
from bcd import bottleCapDetect


class dropLabel(QtWidgets.QLabel):
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    dropDone = QtCore.pyqtSignal(str)

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            self.setPixmap(
                QtGui.QPixmap(f).scaled(self.width(), self.height()))
            self.dropDone.emit(f)


class Ui_MainWindow:
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1204, 667)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("image/tunnel.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        # self.inputImage = QtWidgets.QLabel(self.centralwidget)
        self.inputImage = dropLabel(self.centralwidget)
        self.inputImage.setGeometry(QtCore.QRect(20, 60, 481, 401))
        self.inputImage.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.inputImage.setStyleSheet("border-color: rgb(211, 215, 207);")
        self.inputImage.setFrameShape(QtWidgets.QFrame.Box)
        self.inputImage.setText("")
        self.inputImage.setAcceptDrops(True)
        self.inputImage.setObjectName("inputImage")
        self.inputImage.dropDone.connect(self.setButt)
        self.outputImage = QtWidgets.QLabel(self.centralwidget)
        self.outputImage.setGeometry(QtCore.QRect(660, 60, 481, 401))
        self.outputImage.setFrameShape(QtWidgets.QFrame.Box)
        self.outputImage.setText("")
        self.outputImage.setObjectName("outputImage")
        self.startBt = QtWidgets.QPushButton(self.centralwidget)
        self.startBt.setGeometry(QtCore.QRect(520, 370, 121, 25))
        self.startBt.setObjectName("startBt")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1124, 22))
        self.menubar.setStyleSheet("background-color: rgb(211, 215, 207);\n"
                                   "border-bottom-color: rgb(46, 52, 54);")
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.addAct = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("image/file.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addAct.setIcon(icon1)
        self.addAct.setObjectName("addAct")
        self.addAct.triggered.connect(self.setImage)
        self.quitAct = QtWidgets.QAction(MainWindow)
        self.rmAct = QtWidgets.QAction("&remove image", MainWindow)
        self.rmAct.setStatusTip("remove the input image")
        self.rmAct.setIcon(QtGui.QIcon("image/rm.svg"))
        self.rmAct.triggered.connect(self.removeImage)
        self.saveAct = QtWidgets.QAction("&save the output", MainWindow)
        self.saveAct.setStatusTip("save the output image")
        self.saveAct.setIcon(QtGui.QIcon("image/save.svg"))
        self.saveAct.setShortcut("ctrl+alt+s")
        self.saveAct.triggered.connect(self.save)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("image/quit.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.quitAct.setIcon(icon2)
        self.quitAct.setObjectName("quitAct")
        self.quitAct.triggered.connect(QtWidgets.QApplication.instance().exit)
        self.menuFile.addAction(self.addAct)
        self.menuFile.addAction(self.rmAct)
        self.menuFile.addAction(self.saveAct)
        self.menuFile.addAction(self.quitAct)
        self.saveAct.setEnabled(False)
        self.rmAct.setEnabled(False)
        self.menubar.addAction(self.menuFile.menuAction())
        self.startBt.setEnabled(False)
        self.startBt.clicked.connect(self.start)
        self.retranslateUi(MainWindow)
        # thread
        self.thread = QtCore.QThread()
        self.obj = Calculate()
        self.obj.outReady.connect(self.onOutReady)
        self.obj.moveToThread(self.thread)
        self.obj.finished.connect(self.thread.quit)
        self.thread.started.connect(self.obj.conv)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def onOutReady(self, output):
        self.out = output.__deepcopy__(output)
        self.saveAct.setEnabled(True)
        if output.ndim == 2:
            qim = qimage2ndarray.gray2qimage(output)
        else:
            qim = qimage2ndarray.array2qimage(output)
        self.outputImage.setPixmap(
            QtGui.QPixmap(qim).scaled(self.outputImage.width(), self.outputImage.height()))
        self.startBt.setEnabled(True)
        self.centralwidget.setEnabled(True)

    def setButt(self, f):
        self.ndimage = imread(f)
        self.startBt.setEnabled(True)
        self.rmAct.setEnabled(True)
        self.outputImage.clear()

    def save(self):
        file = QtWidgets.QFileDialog.getSaveFileName(self.centralwidget, "Save Image", ".jpg",
                                                     "JPEG Image (*.jpg);;PNG Image (*.png)")
        if file[0]:
            if self.out.ndim > 2:
                success = imsave(file[0], self.out)
            else:
                success = imwrite(file[0], self.out * 256)
            if success:
                QtWidgets.QMessageBox.information(self.centralwidget, "Succeed!",
                                                  "Image has been successfully saved as {}!\n".format(file[0]))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bottle Cap Detection"))
        self.startBt.setText(_translate("MainWindow", "start"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.addAct.setText(_translate("MainWindow", "Add image"))
        self.addAct.setStatusTip(_translate("MainWindow", "Add/change the image"))
        self.addAct.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.quitAct.setText(_translate("MainWindow", "Quit"))
        self.quitAct.setStatusTip(_translate("MainWindow", "Exit the appliction"))
        self.quitAct.setShortcut(_translate("MainWindow", "Ctrl+Q"))

    def start(self):
        self.startBt.setEnabled(False)
        self.centralwidget.setEnabled(False)
        self.centralwidget.update()
        self.obj.ndimage = self.ndimage
        self.thread.start()

    def removeImage(self):
        if self.inputImage.pixmap():
            self.inputImage.clear()
            self.startBt.setEnabled(False)
            self.outputImage.clear()
            self.saveAct.setEnabled(False)
            self.rmAct.setEnabled(False)

    def setImage(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, 'Open file', '/home',
                                                      "Image Files(*.jpg *.png *.jpeg *.bmp)")
        if fname[0]:
            self.inputImage.setPixmap(
                QtGui.QPixmap(fname[0]).scaled(self.inputImage.width(), self.inputImage.height()))
            self.ndimage = imread(fname[0])
            self.startBt.setEnabled(True)
            self.rmAct.setEnabled(True)
            self.outputImage.clear()


class Calculate(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    outReady = QtCore.pyqtSignal(np.ndarray)
    ndimage = None

    @QtCore.pyqtSlot()
    def conv(self):
        output = bottleCapDetect(self.ndimage)
        self.outReady.emit(output)
        self.finished.emit()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
