import time
import os
import fnmatch
import shutil
import random
import pathlib
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QPixmap
import sys
import numpy as np
from PIL.ImageQt import ImageQt
from PIL import Image
import cv2

from settings1 import Ui_Settings, Comm


class Ui_Dialog(object):
    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Settings()
        self.ui.setupUi(self.window)
        self.ui.set_fields(self.kernelSize, self.input2,
                           self.input3, self.input4)
        self.comm = Comm()
        self.comm.submitted.connect(self.updateSettings)
        self.window.show()

    def setupUi(self, Dialog):

        self.kernelSize = 3
        self.input2 = 9
        self.input3 = 75
        self.input4 = 75
        self.kernel_vals = "0, 0, 0, 0, 0, 0, 0, 0, 0"
        self.manKernel = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        Dialog.setObjectName("Dialog")
        Dialog.resize(1134, 749)

        self.RenderButton = QtWidgets.QPushButton(
            Dialog, clicked=lambda: self.render_start()
        )
        self.RenderButton.setGeometry(QtCore.QRect(10, 700, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Fira Code")
        font.setPointSize(14)
        self.RenderButton.setFont(font)
        self.RenderButton.setObjectName("RenderButton")
        self.ProjectTitle = QtWidgets.QLabel(Dialog)
        self.ProjectTitle.setGeometry(QtCore.QRect(20, -10, 551, 71))
        font = QtGui.QFont()
        font.setFamily("Fira Code")
        font.setPointSize(22)
        self.ProjectTitle.setFont(font)
        self.ProjectTitle.setObjectName("ProjectTitle")

        self.browse_but = QtWidgets.QPushButton(Dialog)
        self.browse_but.setGeometry(QtCore.QRect(400, 710, 93, 28))
        self.browse_but.setObjectName("browse_but")
        self.browse_but.clicked.connect(self.browser)

        self.imagelabel = QtWidgets.QLabel(Dialog)
        self.imagelabel.setGeometry(QtCore.QRect(70, 70, 1011, 611))
        self.imagelabel.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.imagelabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.imagelabel.setObjectName("imagelabel")
        self.blur_but = QtWidgets.QComboBox(Dialog)
        self.blur_but.setGeometry(QtCore.QRect(930, 10, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Fira Code")
        font.setPointSize(10)
        self.blur_but.setFont(font)
        self.blur_but.setObjectName("blur_but")
        self.blur_but.addItem("")
        self.blur_but.addItem("")
        self.blur_but.addItem("")
        self.blur_but.addItem("")
        self.blur_but.addItem("")
        self.smooth_but = QtWidgets.QComboBox(Dialog)
        self.smooth_but.setGeometry(QtCore.QRect(800, 10, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Fira Code")
        font.setPointSize(10)
        self.smooth_but.setFont(font)
        self.smooth_but.setObjectName("smooth_but")
        self.smooth_but.addItem("")
        self.smooth_but.addItem("")
        self.smooth_but.addItem("")
        self.smooth_but.addItem("")
        self.smooth_but.addItem("")
        self.noise_check = QtWidgets.QCheckBox(Dialog)
        self.noise_check.setGeometry(QtCore.QRect(140, 710, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Fira Code")
        font.setPointSize(10)
        self.noise_check.setFont(font)
        self.noise_check.setObjectName("noise_check")

        ################### Mode ####################
        self.pushButton = QtWidgets.QPushButton(
            Dialog, clicked=lambda: self.mode_switch()
        )
        self.pushButton.setGeometry(QtCore.QRect(590, 10, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Fira Code")
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.sharpen = True
        ################### Mode ####################

        self.settings_but = QtWidgets.QPushButton(
            Dialog, clicked=lambda: self.openWindow()
        )
        self.settings_but.setGeometry(QtCore.QRect(280, 710, 93, 28))
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(
                "../../../../../Users/Lenovo/Downloads/Windows_Settings_app_icon.png"
            ),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.Off,
        )
        self.settings_but.setIcon(icon)
        self.settings_but.setObjectName("settings_but")

        self.reset_btn = QtWidgets.QPushButton(Dialog)
        self.reset_btn.setGeometry(QtCore.QRect(780, 710, 93, 28))
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.clicked.connect(self.reset_img)

        self.save_btn = QtWidgets.QPushButton(Dialog)
        self.save_btn.setGeometry(QtCore.QRect(930, 710, 93, 28))
        self.save_btn.setObjectName("save_btn")
        self.save_btn.clicked.connect(self.saveas)

        self.peek_btn = QtWidgets.QPushButton(Dialog)
        self.peek_btn.setGeometry(QtCore.QRect(1030, 710, 93, 28))
        self.peek_btn.setObjectName("peek_btn")
        self.peek_btn.pressed.connect(self.peek_press)
        self.peek_btn.released.connect(self.peek_release)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate(
            "Dialog", "Name: Birinder Singh, Roll No: 102003247"))
        self.RenderButton.setText(_translate("Dialog", "Render"))
        self.ProjectTitle.setText(_translate(
            "Dialog", "OpenCV Image Smoothener"))
        self.browse_but.setText(_translate("Dialog", "Browse"))
        self.imagelabel.setText(_translate("Dialog", "Browse for an Image"))
        self.blur_but.setItemText(0, _translate("Dialog", "Blur"))
        self.blur_but.setItemText(1, _translate("Dialog", "Gaussian Blur"))
        self.blur_but.setItemText(2, _translate("Dialog", "Median Blur"))
        self.blur_but.setItemText(3, _translate("Dialog", "Bilateral Filter"))
        self.blur_but.setItemText(
            4, _translate("Dialog", "Fast Nl Means Denoising Colored")
        )
        self.smooth_but.setItemText(0, _translate("Dialog", "Sharp 1"))
        self.smooth_but.setItemText(1, _translate("Dialog", "Sharp 2"))
        self.smooth_but.setItemText(2, _translate("Dialog", "Sharp 3"))
        self.smooth_but.setItemText(3, _translate("Dialog", "Sharp 4"))
        self.smooth_but.setItemText(4, _translate("Dialog", "Sharp 5"))
        self.noise_check.setText(_translate("Dialog", "Add Noise"))
        self.pushButton.setText(_translate("Dialog", "Mode: Sharpen"))
        self.settings_but.setText(_translate("Dialog", "Settings"))
        self.reset_btn.setText(_translate("Dialog", "Reset"))
        self.save_btn.setText(_translate("Dialog", "Save"))
        self.peek_btn.setText(_translate("Dialog", "Peek"))

    # @QtCore.pyqtSlot(str,str,str,str,str)
    def updateSettings(self, kernel, i2, i3, i4, kvals):
        self.kernelSize = kernel
        self.input2 = i2
        self.input3 = i3
        self.input4 = i4
        self.kernel_vals = kvals
        self.kernel_ints = map(int, kvals.split(","))
        self.manKernel = [[0 for i in range(3)] for j in range(3)]
        j = -1
        for i, item in enumerate(self.kernel_ints):
            if i % 3 == 0:
                j += 1
            self.manKernel[j][i] = item

    def qtToCvPath(self):
        p = pathlib.PureWindowsPath(f"{self.imagepath}")
        return str(p.as_posix())

    def saveas(self):
        try:
            t = time.localtime()
            timestamp = time.strftime("%b-%d-%Y_%H%M", t)
            filename = "Export-" + timestamp + ".jpg"
            self.curr_img_qt.save(
                "/Birinder/Programs/Python/Edge AI/Project/Exports/"+f"{filename}", "JPG")
            print(f"{filename}")
        except:
            print("Save Failed")

    def reset_img(self):
        try:
            pixmap = QPixmap(self.imagepath)
            self.imagelabel.setPixmap(QPixmap(pixmap))
            self.curr_img_qt = pixmap
            pixmap.save("temp.jpg", "JPG")
        except:
            pass

    def peek_press(self):
        try:
            self.imagelabel.setPixmap(QPixmap(self.imagepath))
        except:
            pass

    def peek_release(self):
        pixmap = QPixmap(self.curr_img_qt)
        self.imagelabel.setPixmap(QPixmap(pixmap))

    def browser(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open Photo", "C:\Birinder\Programs\Python\Edge AI\Images"
        )
        self.imagepath = fname[0]
        self.cvpath = self.qtToCvPath()
        pixmap = QPixmap(self.imagepath)
        self.imagelabel.setPixmap(QPixmap(pixmap))
        self.curr_img_qt = pixmap
        pixmap.save("temp.jpg", "JPG")

    def mode_switch(self):
        if self.pushButton.sharpen == True:
            self.pushButton.setText("Mode: Blur")
        else:
            self.pushButton.setText("Mode: Sharpen")

        self.pushButton.sharpen = not self.pushButton.sharpen

    @staticmethod
    def QPixmapToArray(pixmap):
        # Get the size of the current pixmap
        size = pixmap.size()
        h = size.width()
        w = size.height()

        # Get the QImage Item and convert it to a byte string
        qimg = pixmap.toImage()
        byte_str = qimg.bits().tobytes()

        # Using the np.frombuffer function to convert the byte string into an np array
        img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))

        return img

    # @staticmethod
    # def QPixmapToArray(pixmap):
    #     channels_count = 4
    #     image = Image.fromarray(pixmap)
    #     # image = pixmap.toImage()
    #     s = image.bits().asstring(self._width * self._height * channels_count)
    #     arr = np.fromstring(s, dtype=np.uint8).reshape(
    #         (self._height, self._width, channels_count)
    #     )

    @staticmethod
    def convertCvImage2QtImage(cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(rgb_image).convert("RGB")
        return QPixmap.fromImage(ImageQt(PIL_image))

    def render_start(self):

        # img = self.QPixmapToArray(self.curr_img_qt)
        img = cv2.imread("temp.jpg")
        img = img[:, :, ::-1]

        #############################

        def smoothsharpenImage(image, smoothType, kernelSize, input2, input3, input4):

            # Smoothing
            if smoothType == 0:
                return cv2.blur(image, (kernelSize, kernelSize))
            elif smoothType == 1:
                return cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
            elif smoothType == 2:
                return cv2.medianBlur(image, kernelSize)
            elif smoothType == 3:
                return cv2.bilateralFilter(
                    image, input2, input3, input3
                )  # d, sigmaColor, sigmaSpace eg. 9,75,75
            elif smoothType == 4:
                return cv2.fastNlMeansDenoisingColored(
                    image, None, input2, input2, input3, input4
                )  # h, templateWindowSize, searchWindowSize eg. 10,7,21

            # Sharpening
            elif smoothType == 5:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                return cv2.filter2D(image, -1, kernel)
            elif smoothType == 6:
                kernel = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
                return cv2.filter2D(image, -1, kernel)
            elif smoothType == 7:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                return cv2.filter2D(image, -1, kernel)
            elif smoothType == 8:
                kernel = np.array(
                    [
                        [-1, -1, -1, -1, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, 2, 8, 2, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, -1, -1, -1, -1],
                    ]
                ) * (1.0 / 8.0)
                return cv2.filter2D(image, -1, kernel)
            elif smoothType == 9:
                kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
                return cv2.filter2D(image, -1, kernel)

        def addSaltPepperNoise(image, y, x, h, w, intensity):  # column first row second
            copyimg = image.copy()
            nop = random.randint(5000, 1000 * intensity)
            applysalt = True
            rows, cols = copyimg.shape[:2]
            for i in range(nop):
                m = random.randint(0, copyimg.shape[0] - 1)
                n = random.randint(0, copyimg.shape[1] - 1)
                #     # if(TRUE):# (m<x or m>x+w )or (n<y or n>y+h)):
                if applysalt:
                    copyimg[m, n] = [255, 255, 255]  # salt
                else:
                    copyimg[m, n] = [0, 0, 0]  # pepper
                applysalt = not applysalt
            return copyimg

        ###############################

        if self.noise_check.isChecked():
            retcv = addSaltPepperNoise(img, 0, 0, 0, 0, 10)
            # retcv.save("temp.jpg")
            # retcv = retcv[:, :, ::-1]
            retcv = Image.fromarray(retcv)
            retcv.save("temp.jpg")
            pixmap = QPixmap("temp.jpg")
            self.imagelabel.setPixmap(QPixmap(pixmap))
            self.curr_img_qt = pixmap
            return

        if self.pushButton.sharpen:
            smoothType = self.smooth_but.currentIndex() + 5

        else:
            smoothType = self.blur_but.currentIndex()

        retcv = smoothsharpenImage(
            img, smoothType, self.kernelSize, self.input2, self.input3, self.input4
        )
        retcv = Image.fromarray(retcv)
        retcv.save("temp.jpg")
        pixmap = QPixmap("temp.jpg")
        self.imagelabel.setPixmap(QPixmap(pixmap))
        self.curr_img_qt = pixmap
        return


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
