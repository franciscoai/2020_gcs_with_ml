# from GCSgui import Ui_MainWindow  # importing our generated file
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pyGCS import *

global nSats
global CMElat, CMElon, CMEtilt, height, k, ang, satpos
global fname

# This will output the values to this filename so it can reload
# after it has been closed.  Delete this file to open with defaults
fname = 'GCSvals.txt'


class Ui_MainWindow(object):
    # This generically sets up the main components of the GUI (was largely
    # produced using the pyqt developer)
    def setupUi(self, MainWindow, nSats):
        # Set up the main window and its properties
        MainWindow.setObjectName("MainWindow")
        wsize = 375
        MainWindow.resize(260+wsize*nSats, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # Set up one to three plotting windows depending on nSats -----------------------|
        ypos = int((600-wsize)/2)-75
        self.graphWidget1 = pg.PlotWidget(self.centralwidget)
        self.graphWidget1.setGeometry(QtCore.QRect(220, ypos, wsize, wsize))
        self.labelSat1 = QtWidgets.QLabel(self.centralwidget)
        self.labelSat1.setGeometry(QtCore.QRect(220, 20, 150, 16))
        if nSats >= 2:
            self.graphWidget2 = pg.PlotWidget(self.centralwidget)
            self.graphWidget2.setGeometry(QtCore.QRect(230+wsize, ypos, wsize, wsize))
            self.labelSat2 = QtWidgets.QLabel(self.centralwidget)
            self.labelSat2.setGeometry(QtCore.QRect(230+wsize, 20, 150, 16))
        if nSats == 3:
            self.graphWidget3 = pg.PlotWidget(self.centralwidget)
            self.graphWidget3.setGeometry(QtCore.QRect(240+2*wsize, ypos, wsize, wsize))
            self.labelSat3 = QtWidgets.QLabel(self.centralwidget)
            self.labelSat3.setGeometry(QtCore.QRect(240+2*wsize, 20, 150, 16))

        # Set up the individual sliders, their text boxes, and their labels -------------|
        # GCS shell parameters here
        # Latitude
        self.sliderLat = QtWidgets.QSlider(self.centralwidget)
        self.sliderLat.setGeometry(QtCore.QRect(30, 130, 160, 22))
        self.sliderLat.setOrientation(QtCore.Qt.Horizontal)
        self.sliderLat.setMinimum(-90)
        self.sliderLat.setMaximum(90)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 90, 59, 16))
        self.leLat = QtWidgets.QLineEdit(self.centralwidget)
        self.leLat.setGeometry(QtCore.QRect(30, 110, 113, 21))
        self.leLat.setText('0')         # set to a default value, may get replaced
        # Longitude
        self.sliderLon = QtWidgets.QSlider(self.centralwidget)
        self.sliderLon.setGeometry(QtCore.QRect(30, 60, 160, 22))
        self.sliderLon.setOrientation(QtCore.Qt.Horizontal)
        self.sliderLon.setMinimum(-180)
        self.sliderLon.setMaximum(180)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 20, 71, 16))
        self.leLon = QtWidgets.QLineEdit(self.centralwidget)
        self.leLon.setGeometry(QtCore.QRect(30, 40, 113, 21))
        self.leLon.setText('0')
        # Tilt
        self.sliderTilt = QtWidgets.QSlider(self.centralwidget)
        self.sliderTilt.setGeometry(QtCore.QRect(30, 200, 160, 22))
        self.sliderTilt.setOrientation(QtCore.Qt.Horizontal)
        self.sliderTilt.setMinimum(-90)
        self.sliderTilt.setMaximum(90)
        self.sliderTilt.setValue(10)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 160, 71, 16))
        self.leTilt = QtWidgets.QLineEdit(self.centralwidget)
        self.leTilt.setGeometry(QtCore.QRect(30, 180, 113, 21))
        self.leTilt.setText('10')
        # Height
        self.sliderHeight = QtWidgets.QSlider(self.centralwidget)
        self.sliderHeight.setGeometry(QtCore.QRect(30, 270, 160, 22))
        self.sliderHeight.setOrientation(QtCore.Qt.Horizontal)
        self.sliderHeight.setMinimum(11)
        self.sliderHeight.setMaximum(250)
        self.sliderHeight.setValue(50)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 230, 71, 16))
        self.leHeight = QtWidgets.QLineEdit(self.centralwidget)
        self.leHeight.setGeometry(QtCore.QRect(30, 250, 113, 21))
        self.leHeight.setText('5.0')
        # Angular Width
        self.sliderAW = QtWidgets.QSlider(self.centralwidget)
        self.sliderAW.setGeometry(QtCore.QRect(30, 340, 160, 22))
        self.sliderAW.setOrientation(QtCore.Qt.Horizontal)
        self.sliderAW.setMinimum(5)
        self.sliderAW.setMaximum(90)
        self.sliderAW.setValue(30)
        self.leAW = QtWidgets.QLineEdit(self.centralwidget)
        self.leAW.setGeometry(QtCore.QRect(30, 320, 113, 21))
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 300, 91, 16))
        self.leAW.setText('30')
        # Kappa/ratio
        self.sliderK = QtWidgets.QSlider(self.centralwidget)
        self.sliderK.setGeometry(QtCore.QRect(30, 420, 160, 22))
        self.sliderK.setOrientation(QtCore.Qt.Horizontal)
        self.sliderK.setMinimum(5)
        self.sliderK.setMaximum(90)
        self.sliderK.setValue(20)
        self.leK = QtWidgets.QLineEdit(self.centralwidget)
        self.leK.setGeometry(QtCore.QRect(30, 400, 113, 21))
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 380, 71, 16))
        self.leK.setText('0.20')

        # Label that originally was for saving but now just tells user that -------------|
        # you only need to right click for built in saving
        self.saveLabel = QtWidgets.QLabel(self.centralwidget)
        self.saveLabel.setGeometry(QtCore.QRect(30, 520, 113, 32))

        # Button for turining the wireframe on or off -----------------------------------|
        self.wireButton = QtWidgets.QPushButton(self.centralwidget)
        self.wireButton.setGeometry(QtCore.QRect(30, 440, 131, 32))

        # Sliders and drop menu for scaling parameters ----------------------------------|
        # Matches number of plot windows
        # Drop menu
        self.menuScale = QtWidgets.QComboBox(self.centralwidget)
        self.menuScale.setGeometry(QtCore.QRect(30, 490, 90, 22))
        self.menuScale.addItems(["Linear", "Log", "Sqrt"])
        # Sat1 minimum brightness
        self.slSat1low = QtWidgets.QSlider(self.centralwidget)
        self.slSat1low.setGeometry(QtCore.QRect(220, ypos+wsize+50, 160, 22))
        self.slSat1low.setOrientation(QtCore.Qt.Horizontal)
        self.leSat1low = QtWidgets.QLineEdit(self.centralwidget)
        self.leSat1low.setGeometry(QtCore.QRect(220, ypos+wsize+30, 160, 22))
        self.labelSat1low = QtWidgets.QLabel(self.centralwidget)
        self.labelSat1low.setGeometry(QtCore.QRect(220, ypos+wsize+10, 160, 22))
        # Sat1 maximum brightness
        self.slSat1hi = QtWidgets.QSlider(self.centralwidget)
        self.slSat1hi.setGeometry(QtCore.QRect(220, ypos+wsize+120, 160, 22))
        self.slSat1hi.setOrientation(QtCore.Qt.Horizontal)
        self.leSat1hi = QtWidgets.QLineEdit(self.centralwidget)
        self.leSat1hi.setGeometry(QtCore.QRect(220, ypos+wsize+100, 160, 22))
        self.labelSat1hi = QtWidgets.QLabel(self.centralwidget)
        self.labelSat1hi.setGeometry(QtCore.QRect(220, ypos+wsize+80, 160, 22))
        if nSats > 1:
            # Sat2 minimum brightness
            self.slSat2low = QtWidgets.QSlider(self.centralwidget)
            self.slSat2low.setGeometry(QtCore.QRect(220+wsize+10, ypos+wsize+50, 160, 22))
            self.slSat2low.setOrientation(QtCore.Qt.Horizontal)
            self.leSat2low = QtWidgets.QLineEdit(self.centralwidget)
            self.leSat2low.setGeometry(QtCore.QRect(220+wsize+10, ypos+wsize+30, 160, 22))
            self.labelSat2low = QtWidgets.QLabel(self.centralwidget)
            self.labelSat2low.setGeometry(QtCore.QRect(220+wsize+10, ypos+wsize+10, 160, 22))
            # Sat2 maximum brightness
            self.slSat2hi = QtWidgets.QSlider(self.centralwidget)
            self.slSat2hi.setGeometry(QtCore.QRect(220+wsize+10, ypos+wsize+120, 160, 22))
            self.slSat2hi.setOrientation(QtCore.Qt.Horizontal)
            self.leSat2hi = QtWidgets.QLineEdit(self.centralwidget)
            self.leSat2hi.setGeometry(QtCore.QRect(220+wsize+10, ypos+wsize+100, 160, 22))
            self.labelSat2hi = QtWidgets.QLabel(self.centralwidget)
            self.labelSat2hi.setGeometry(QtCore.QRect(220+wsize+10, ypos+wsize+80, 160, 22))
        if nSats == 3:
            # Sat3 minimum brightness
            self.slSat3low = QtWidgets.QSlider(self.centralwidget)
            self.slSat3low.setGeometry(QtCore.QRect(220+2*wsize+20, ypos+wsize+50, 160, 22))
            self.slSat3low.setOrientation(QtCore.Qt.Horizontal)
            self.leSat3low = QtWidgets.QLineEdit(self.centralwidget)
            self.leSat3low.setGeometry(QtCore.QRect(220+2*wsize+20, ypos+wsize+30, 160, 22))
            self.labelSat3low = QtWidgets.QLabel(self.centralwidget)
            self.labelSat3low.setGeometry(QtCore.QRect(220+2*wsize+20, ypos+wsize+10, 160, 22))
            # Sat3 maximum brightness
            self.slSat3hi = QtWidgets.QSlider(self.centralwidget)
            self.slSat3hi.setGeometry(QtCore.QRect(220+2*wsize+20, ypos+wsize+120, 160, 22))
            self.slSat3hi.setOrientation(QtCore.Qt.Horizontal)
            self.leSat3hi = QtWidgets.QLineEdit(self.centralwidget)
            self.leSat3hi.setGeometry(QtCore.QRect(220+2*wsize+20, ypos+wsize+100, 160, 22))
            self.labelSat3hi = QtWidgets.QLabel(self.centralwidget)
            self.labelSat3hi.setGeometry(QtCore.QRect(220+2*wsize+20, ypos+wsize+80, 160, 22))

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow, nSats)

    def retranslateUi(self, MainWindow, nSats):
        # This takes the generic widgets and renames them what we want -----------------|
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "pyGCS"))
        self.label.setText(_translate("MainWindow", "Latitude"))
        self.label_2.setText(_translate("MainWindow", "Longitude"))
        self.label_3.setText(_translate("MainWindow", "Tilt"))
        self.label_4.setText(_translate("MainWindow", "Height"))
        self.label_5.setText(_translate("MainWindow", "Half Angle"))
        self.label_6.setText(_translate("MainWindow", "Ratio"))
        self.saveLabel.setText(_translate("MainWindow", "Right click to save"))
        self.wireButton.setText(_translate("MainWindow", "Wireframe On/Off"))
        self.labelSat1.setText(_translate("MainWindow", "Sat 1"))
        self.labelSat1low.setText(_translate("MainWindow", "Min Brightness"))
        self.labelSat1hi.setText(_translate("MainWindow", "Max Brightness"))
        if nSats > 1:
            self.labelSat2.setText(_translate("MainWindow", "Sat 2"))
            self.labelSat2low.setText(_translate("MainWindow", "Min Brightness"))
            self.labelSat2hi.setText(_translate("MainWindow", "Max Brightness"))
        if nSats == 3:
            self.labelSat3.setText(_translate("MainWindow", "Sat 3"))
            self.labelSat3low.setText(_translate("MainWindow", "Sat 3 Min Brightness"))
            self.labelSat3hi.setText(_translate("MainWindow", "Sat 3 Max Brightness"))


class mywindow(QtWidgets.QMainWindow):
    # This takes the generic but properly labeled window and adapts it to ---------------|
    # our specific needs
    def __init__(self, imsIn, satposIn, plotrangesIn, sats, nsIn=[5, 20, 30]):
        # Set up globals for the number of sats, plotranges, original images
        # the actual images displayed in the GUI, and the wireframe point density
        global nSats, plotranges, imgOrig, imgOut, ns
        nSats = len(satposIn)
        plotranges = plotrangesIn
        imgOrig = []
        imgOut = []
        for im in imsIn:
            # Convenient trick to put LASCO images in a
            # more similar range to STEREO images
            if (np.median(np.abs(im))) < 1e-5:
                im = im / (np.median(np.abs(im)))
            imgOrig.append(im)
            imgOut.append(im)
        ns = nsIn

        # -------------------------------------------------------------------------------|
        # ESSENTIAL GUI SETUP THAT I TOTALLY UNDERSTAND! --------------------------------|
        super(mywindow, self).__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        # -------------------------------------------------------------------------------|
        # Get a generic MainWindow then add our labels ----------------------------------|
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self, nSats)

        # -------------------------------------------------------------------------------|
        # Give nice titles (sat+instr) to each plot -------------------------------------|
        for i in range(nSats):
            if i == 0:
                self.ui.labelSat1.setText(sats[i][0]+' '+sats[i][1])
            if i == 1:
                self.ui.labelSat2.setText(sats[i][0]+' '+sats[i][1])
            if i == 2:
                self.ui.labelSat3.setText(sats[i][0]+' '+sats[i][1])

        # -------------------------------------------------------------------------------|
        # Take the input sat info and assign to global ----------------------------------|
        # and make an array to map GCS scatter into
        # image coords
        global satpos, scaleNshift
        satpos = satposIn
        scaleNshift = []
        # Currently lazy, assuming x=y and actually centered on sun
        # but could pass in header information to calc?
        for i in range(nSats):
            scaleNshift.append([imsIn[i].shape[0]/(2*plotranges[i][1]), int(imsIn[i].shape[0]/2)])

        # -------------------------------------------------------------------------------|
        # Make a mask for the occulter and outside circular FOV -------------------------|
        # Again being lazy and making the same Sun centered approx
        global masks, innerR
        # Occulter distance for each satellite
        instrDict = {'COR1': 1.3, 'COR2': 2, 'C2': 1.5, 'C3': 3.7}
        masks = []
        innerR = []
        cents = []
        for idx in range(nSats):
            myInst = sats[idx][1]
            mask = np.zeros(imsIn[idx].shape)
            cent = int(imsIn[idx].shape[0]/2)
            line = np.linspace(0, imsIn[idx].shape[1]-1, imsIn[idx].shape[1])
            for i in range(imsIn[idx].shape[0]):
                mask[i, :] = np.sqrt((i-cent)**2 + (line-cent)**2)
            mask = mask / scaleNshift[idx][0]
            mask[np.where(mask > plotrangesIn[idx][1])] = 1.
            mask[np.where(mask < instrDict[myInst])] = 1.
            innerR.append(instrDict[myInst])
            masks.append(mask)

        # -------------------------------------------------------------------------------|
        # Set up the image spots in the GUI and make an array ---------------------------|
        # holding the graphWidgets so we can access elsewhere
        global images
        images = []
        image1 = pg.ImageItem()
        self.ui.graphWidget1.addItem(image1)
        self.ui.graphWidget1.setRange(xRange=(0, imsIn[0].shape[0]), yRange=(0, imsIn[0].shape[0]), padding=0)
        self.ui.graphWidget1.hideAxis('bottom')
        self.ui.graphWidget1.hideAxis('left')
        images.append(image1)
        if nSats > 1:
            image2 = pg.ImageItem()
            self.ui.graphWidget2.addItem(image2)
            self.ui.graphWidget2.setRange(xRange=(0, imsIn[1].shape[0]), yRange=(0, imsIn[1].shape[0]), padding=0)
            self.ui.graphWidget2.hideAxis('bottom')
            self.ui.graphWidget2.hideAxis('left')
            images.append(image2)
        if nSats == 3:
            image3 = pg.ImageItem()
            self.ui.graphWidget3.addItem(image3)
            self.ui.graphWidget3.setRange(xRange=(0, imsIn[2].shape[0]), yRange=(0, imsIn[2].shape[0]), padding=0)
            self.ui.graphWidget3.hideAxis('bottom')
            self.ui.graphWidget3.hideAxis('left')
            images.append(image3)

        # -------------------------------------------------------------------------------|
        # Check if there is an file with previous values to load ------------------------|
        minmaxesIn = [[-9999, -9999], [-9999, -9999], [-9999, -9999]]
        if (os.path.exists(fname)):
            minmaxesIn = self.initShellValues()

        # -------------------------------------------------------------------------------|
        # Initialize CME variables to the defaults or whatever was ----------------------|
        # pulled in from the input file
        global CMElat, CMElon, CMEtilt, height, k, ang, satpo
        CMElon = float(self.ui.leLon.text())
        CMElat = float(self.ui.leLat.text())
        CMEtilt = float(self.ui.leTilt.text())
        height = float(self.ui.leHeight.text())
        ang = float(self.ui.leAW.text())
        k = float(self.ui.leK.text())

        # -------------------------------------------------------------------------------|
        # Connect the sliders and textboxes to their actions (and each other!) ----------|
        self.ui.sliderLon.valueChanged[int].connect(self.slLon)
        self.ui.leLon.returnPressed.connect(self.allGCSText)
        self.ui.sliderLat.valueChanged[int].connect(self.slLat)
        self.ui.leLat.returnPressed.connect(self.allGCSText)
        self.ui.sliderTilt.valueChanged[int].connect(self.slTilt)
        self.ui.leTilt.returnPressed.connect(self.allGCSText)
        self.ui.sliderHeight.valueChanged[int].connect(self.slHeight)
        self.ui.leHeight.returnPressed.connect(self.allGCSText)
        self.ui.sliderAW.valueChanged[int].connect(self.slAW)
        self.ui.leAW.returnPressed.connect(self.allGCSText)
        self.ui.sliderK.valueChanged[int].connect(self.slK)
        self.ui.leK.returnPressed.connect(self.allGCSText)
        self.ui.slSat1low.valueChanged[int].connect(self.slBmin1)
        self.ui.leSat1low.returnPressed.connect(self.allImText)
        self.ui.slSat1hi.valueChanged[int].connect(self.slBmax1)
        self.ui.leSat1hi.returnPressed.connect(self.allImText)
        if nSats > 1:
            self.ui.slSat2low.valueChanged[int].connect(self.slBmin2)
            self.ui.leSat2low.returnPressed.connect(self.allImText)
            self.ui.slSat2hi.valueChanged[int].connect(self.slBmax2)
            self.ui.leSat2hi.returnPressed.connect(self.allImText)
        if nSats == 3:
            self.ui.slSat3low.valueChanged[int].connect(self.slBmin3)
            self.ui.leSat3low.returnPressed.connect(self.allImText)
            self.ui.slSat3hi.valueChanged[int].connect(self.slBmax3)
            self.ui.leSat3hi.returnPressed.connect(self.allImText)

        # -------------------------------------------------------------------------------|
        # Connect the menu Box ----------------------------------------------------------|
        self.ui.menuScale.currentIndexChanged.connect(self.selectionchange)

        # -------------------------------------------------------------------------------|
        # Connect the wireframe button --------------------------------------------------|
        global wireShow
        wireShow = True
        self.ui.wireButton.clicked.connect(self.wireOO)

        # -------------------------------------------------------------------------------|
        # Set up a global for the minmax brightness -------------------------------------|
        # This can either come from the input file or be
        # calculated by initBrange
        global minmaxes
        minmaxes = np.array([[-1, 1], [-1, 1], [-1, 1]])
        if minmaxesIn[0][0] == -9999:
            for i in range(nSats):
                minmaxes[i] = self.initBrange(imgOut[i], i)
        else:
            minmaxes = minmaxesIn
        self.resetBrights(minmaxes)

        # -------------------------------------------------------------------------------|
        # Add the mask to the images, then the images to the ----------------------------|
        # widgets (with appropriate levels)
        for i in range(nSats):
            imgOut[i][np.where(masks[i] == 1)] = np.min(imgOut[i])
            images[i].setImage(imgOut[i], levels=minmaxes[i])

        # -------------------------------------------------------------------------------|
        # Set up the scatter items that will hold the GCS shells ------------------------|
        global scatters
        scatters = []
        scatter1 = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='g'), symbol='o', size=2)
        self.ui.graphWidget1.addItem(scatter1)
        scatters.append(scatter1)
        if nSats > 1:
            scatter2 = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='g'), symbol='o', size=2)
            self.ui.graphWidget2.addItem(scatter2)
            scatters.append(scatter2)
        if nSats == 3:
            scatter3 = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='g'), symbol='o', size=2)
            self.ui.graphWidget3.addItem(scatter3)
            scatters.append(scatter3)

        # -------------------------------------------------------------------------------|
        # Draw a white circle showing the outline of the Sun ----------------------------|
        thetas = np.linspace(0, 2*3.14159)
        self.ui.graphWidget1.plot(scaleNshift[0][0]*np.cos(thetas)+scaleNshift[0]
                                  [1], scaleNshift[0][0]*np.sin(thetas)+scaleNshift[0][1])
        if nSats > 1:
            self.ui.graphWidget2.plot(scaleNshift[1][0]*np.cos(thetas)+scaleNshift[1]
                                      [1], scaleNshift[1][0]*np.sin(thetas)+scaleNshift[1][1])
        if nSats == 3:
            self.ui.graphWidget3.plot(scaleNshift[2][0]*np.cos(thetas)+scaleNshift[2]
                                      [1], scaleNshift[2][0]*np.sin(thetas)+scaleNshift[2][1])

        # -------------------------------------------------------------------------------|
        # Calculate the GCS shell using pyGCS and add to the figures --------------------|
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])

    # -----------------------------------------------------------------------------------|
    # -----------------------------------------------------------------------------------|
    # -----------------------------------------------------------------------------------|
    # -----------------------------------------------------------------------------------|

    # Various functions that init will call ---------------------------------------------|
    # Most actions for sliders and text boxes but a few actual things

    # -----------------------------------------------------------------------------------|

    def initShellValues(self):  # ---------------------------------------------------------|
        # Read in the values from an input value and initiate all the
        # sliders and text boxes at appropriate values
        # Returns the minmax brightness so we can set that later
        f1 = open(fname, 'r')
        data = np.genfromtxt(f1, dtype=None, encoding='utf8')
        minmaxesOut = [[-9999, -9999], [-9999, -9999], [-9999, -9999]]
        for line in data:
            if (line[0] == 'Lon:'):
                self.ui.leLon.setText(str(line[1]))
                self.ui.sliderLon.setValue(line[1])
            if (line[0] == 'Lat:'):
                self.ui.leLat.setText(str(line[1]))
                self.ui.sliderLat.setValue(line[1])
            if (line[0] == 'Tilt:'):
                self.ui.leTilt.setText(str(line[1]))
                self.ui.sliderTilt.setValue(line[1])
            if (line[0] == 'Height:'):
                self.ui.leHeight.setText(str(line[1]))
                self.ui.sliderHeight.setValue(10*line[1])
            if (line[0] == 'HalfAng:'):
                self.ui.leAW.setText(str(line[1]))
                self.ui.sliderAW.setValue(line[1])
            if (line[0] == 'Ratio:'):
                self.ui.leK.setText(str(line[1]))
                self.ui.sliderK.setValue(100*line[1])
            if (line[0] == 'Scaling:'):
                revScDict = {0: 'Linear', 1: 'Log', 2: 'Sqrt'}
                self.ui.menuScale.setCurrentText(revScDict[int(line[1])])
                self.selectionchange()
            if (line[0] == 'Sat1min:'):
                self.ui.leSat1low.setText(str(line[1]))
                self.ui.slSat1low.setValue(line[1])
                minmaxesOut[0][0] = line[1]
            if (line[0] == 'Sat1max:'):
                self.ui.leSat1hi.setText(str(line[1]))
                self.ui.slSat1hi.setValue(line[1])
                minmaxesOut[0][1] = line[1]
            if (line[0] == 'Sat2min:') & (nSats > 1):
                self.ui.leSat2low.setText(str(line[1]))
                self.ui.slSat2low.setValue(line[1])
                minmaxesOut[1][0] = line[1]
            if (line[0] == 'Sat2max:') & (nSats > 1):
                self.ui.leSat2hi.setText(str(line[1]))
                self.ui.slSat2hi.setValue(line[1])
                minmaxesOut[1][1] = line[1]
            if (line[0] == 'Sat3min:') & (nSats == 3):
                self.ui.leSat3low.setText(str(line[1]))
                self.ui.slSat3low.setValue(line[1])
                minmaxesOut[2][0] = line[1]
            if (line[0] == 'Sat3max:') & (nSats == 3):
                self.ui.leSat3hi.setText(str(line[1]))
                self.ui.slSat3hi.setValue(line[1])
                minmaxesOut[2][1] = line[1]
        return minmaxesOut

    # -----------------------------------------------------------------------------------|
    def closeEvent(self, event):  # -------------------------------------------------------#
        # Executed when the window is closed. Saves the current
        # values of each parameter so can easily be reloaded
        global fname
        print('Saving output in GCSvals.txt')
        f1 = open(fname, 'w')
        f1.write('Lon:     '+self.ui.leLon.text()+'\n')
        f1.write('Lat:     '+self.ui.leLat.text()+'\n')
        f1.write('Tilt:    '+self.ui.leTilt.text()+'\n')
        f1.write('Height:  '+self.ui.leHeight.text()+'\n')
        f1.write('HalfAng: '+self.ui.leAW.text()+'\n')
        f1.write('Ratio:   '+self.ui.leK.text()+'\n')
        scDict = {'Linear': '0', 'Log': '1', 'Sqrt': '2'}
        f1.write('Scaling: '+scDict[self.ui.menuScale.currentText()]+'\n')
        f1.write('Sat1min: '+self.ui.leSat1low.text()+'\n')
        f1.write('Sat1max: '+self.ui.leSat1hi.text()+'\n')
        if nSats > 1:
            f1.write('Sat2min: '+self.ui.leSat2low.text()+'\n')
            f1.write('Sat2max: '+self.ui.leSat2hi.text()+'\n')
        if nSats == 3:
            f1.write('Sat3min: '+self.ui.leSat3low.text()+'\n')
            f1.write('Sat3max: '+self.ui.leSat3hi.text()+'\n')
        f1.close()

    # -----------------------------------------------------------------------------------|
    def resetBrights(self, minmaxes):  # --------------------------------------------------|
        # Take the minmax values and rescale the images
        self.ui.slSat1low.setValue(int(minmaxes[0][0]))
        self.ui.leSat1low.setText(str(int(minmaxes[0][0])))
        self.ui.slSat1hi.setValue(int(minmaxes[0][1]))
        self.ui.leSat1hi.setText(str(int(minmaxes[0][1])))
        if nSats > 1:
            self.ui.slSat2low.setValue(int(minmaxes[1][0]))
            self.ui.leSat2low.setText(str(int(minmaxes[1][0])))
            self.ui.slSat2hi.setValue(int(minmaxes[1][1]))
            self.ui.leSat2hi.setText(str(int(minmaxes[1][1])))
        if nSats == 3:
            self.ui.slSat3low.setValue(int(minmaxes[2][0]))
            self.ui.leSat3low.setText(str(int(minmaxes[2][0])))
            self.ui.slSat3hi.setValue(int(minmaxes[2][1]))
            self.ui.leSat3hi.setText(str(int(minmaxes[2][1])))

    # -----------------------------------------------------------------------------------|
    def initBrange(self, imIn, idx):  # --------------------------------------------------|
        # Make a guess at a good range for each plot based on
        # the current scaling method and rescale to that value
        absMed = (np.median(np.abs(imIn)))
        if self.ui.menuScale.currentText() == 'Linear':
            Bmin, Bmax = -10*absMed, 10*absMed
            slLow, slHigh = -30*absMed, 30*absMed
        if self.ui.menuScale.currentText() in ['Sqrt', 'Log']:
            Bmin, Bmax = int(absMed), int(1.25*absMed)
            slLow, slHigh = 0, 3*absMed
        # Figure out which slider and box
        if idx == 0:
            sls = [self.ui.slSat1low, self.ui.slSat1hi]
            les = [self.ui.leSat1low, self.ui.leSat1hi]
        if idx == 1:
            sls = [self.ui.slSat2low, self.ui.slSat2hi]
            les = [self.ui.leSat2low, self.ui.leSat2hi]
        if idx == 2:
            sls = [self.ui.slSat3low, self.ui.slSat3hi]
            les = [self.ui.leSat3low, self.ui.leSat3hi]
        # Reset things
        sls[0].setMinimum(slLow)
        sls[0].setMaximum(slHigh)
        sls[1].setMinimum(slLow)
        sls[1].setMaximum(slHigh)
        sls[0].setValue(Bmin)
        sls[1].setValue(Bmax)
        les[0].setText(str(int(Bmin)))
        les[1].setText(str(int(Bmax)))
        return Bmin, Bmax

    # -----------------------------------------------------------------------------------|
    def selectionchange(self):  # --------------------------------------------------------#
        # When the menu box changes reprocess imgOut from imgOrig
        # however is needed.  Also calls initBrange to attempt to
        # make images pretty again
        global imgOut, imgOrig, images, masks
        for i in range(nSats):
            imgOut[i] = np.copy(imgOrig[i])
            # for Linear only need to copy to orig so nothing else...
            if self.ui.menuScale.currentText() == 'Sqrt':
                medval = (np.median(np.abs(imgOrig[i][np.isreal(imgOrig[i])])))
                imgOut[i][np.isreal(imgOrig[i]) == False] = medval
                imgOut[i] += 30.*medval
                imgOut[i][np.where(imgOut[i] < 0)] = 0
                # spread the sqrt out for more integers for slider
                imgOut[i] = 10*np.sqrt(imgOut[i])
            if self.ui.menuScale.currentText() == 'Log':
                medval = (np.median(np.abs(imgOrig[i][np.isreal(imgOrig[i])])))
                imgOut[i][np.isreal(imgOrig[i]) == False] = medval
                imgOut[i] += 30.*medval
                imgOut[i][np.where(imgOut[i] < 0)] = medval
                # spread out again
                imgOut[i] = 10*np.log(imgOut[i])

            Bmin, Bmax = self.initBrange(imgOut[i], i)
            imgOut[i][np.where(masks[i] == 1)] = np.min(imgOut[i])
            images[i].updateImage(image=imgOut[i], levels=(Bmin, Bmax))

    # -----------------------------------------------------------------------------------|
    def wireOO(self):  # ------------------------------------------------------------------|
        # Turn the wireframe on or off
        global wireShow
        if wireShow:
            for i in range(nSats):
                scatters[i].setData()
            wireShow = False
        else:
            data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
            for i in range(nSats):
                self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])
            wireShow = True

    # -----------------------------------------------------------------------------------|
    def plotGCSscatter(self, scatter, dataIn, sNs, rMin):  # ------------------------------|
        # Take the data from pyGCS, shift and scale to match coronagraph
        # range and put it in a happy pyqt format.  If statement will hide
        # any values behind the Sun to try and help projection confusion
        # Can switch full for loop to single pos = [] line to turn off
        pos = []
        for i in range(dataIn.shape[0]):
            if dataIn[i, 0] > 0:
                pos.append({'pos': [sNs[0]*dataIn[i, 1]+sNs[1], sNs[0]*dataIn[i, 2]+sNs[1]]})
            else:
                if dataIn[i, 1]**2 + dataIn[i, 2]**2 > rMin**2:
                    pos.append({'pos': [sNs[0]*dataIn[i, 1]+sNs[1], sNs[0]*dataIn[i, 2]+sNs[1]]})
        # pos = [{'pos': [sNs[0]*dataIn[i,1]+sNs[1], sNs[0]*dataIn[i,2]+sNs[1]]} for i in range(dataIn.shape[0])]
        scatter.setData(pos)

    # -----------------------------------------------------------------------------------|
    # -----------------------------------------------------------------------------------|
    # -----------------------------------------------------------------------------------|
    # All the slider things -------------------------------------------------------------|
    # brightness minmax for each spacecraft
    def slBmin1(self, value):
        global minmaxes, images, imgOut
        minmaxes[0][0] = value
        self.ui.leSat1low.setText(str(value))
        images[0].updateImage(image=imgOut[0], levels=minmaxes[0])

    def slBmax1(self, value):
        global minmaxes, images, imgOut
        minmaxes[0][1] = value
        self.ui.leSat1hi.setText(str(value))
        images[0].updateImage(image=imgOut[0], levels=minmaxes[0])

    def slBmin2(self, value):
        global minmaxes, images, imgOut
        minmaxes[1][0] = value
        self.ui.leSat2low.setText(str(value))
        images[1].updateImage(image=imgOut[1], levels=minmaxes[1])

    def slBmax2(self, value):
        global minmaxes, images, imgOut
        minmaxes[1][1] = value
        self.ui.leSat2hi.setText(str(value))
        images[1].updateImage(image=imgOut[1], levels=minmaxes[1])

    def slBmin3(self, value):
        global minmaxes, images, imgOut
        minmaxes[2][0] = value
        self.ui.leSat3low.setText(str(value))
        images[2].updateImage(image=imgOut[2], levels=minmaxes[2])

    def slBmax3(self, value):
        global minmaxes, images, imgOut
        minmaxes[2][1] = value
        self.ui.leSat3hi.setText(str(value))
        images[2].updateImage(image=imgOut[2], levels=minmaxes[2])

    # Wireframe values
    def slLon(self, value):
        global CMElon
        CMElon = value
        self.ui.leLon.setText(str(CMElon))
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])

    def slLat(self, value):
        global CMElat
        CMElat = value
        self.ui.leLat.setText(str(CMElat))
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])

    def slTilt(self, value):
        global CMEtilt
        CMEtilt = value
        self.ui.leTilt.setText(str(CMEtilt))
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])

    def slHeight(self, value):
        global height
        # scale the height to a larger range to make the slider
        # have better resolution
        height = 0.1*value
        self.ui.leHeight.setText('{:6.2f}'.format(height))
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])

    def slAW(self, value):
        global ang
        ang = value
        self.ui.leAW.setText(str(ang))
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])

    def slK(self, value):
        global k
        # scale the ratio to a larger range to make the slider
        # have better resolution
        k = 0.01*value
        self.ui.leK.setText('{:3.2f}'.format(k))
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])

    # All the text things ---------------------------------------------------------------|
    def allImText(self):
        global minmaxes, images, imgOut, CMElon, CMElat, CMEtilt, height, ang, k
        # Pull all the brightness values
        minmaxes[0][0] = float(self.ui.leSat1low.text())
        minmaxes[0][1] = float(self.ui.leSat1hi.text())
        minmaxes[1][0] = float(self.ui.leSat2low.text())
        minmaxes[1][1] = float(self.ui.leSat2hi.text())
        minmaxes[2][0] = float(self.ui.leSat3low.text())
        minmaxes[2][1] = float(self.ui.leSat3hi.text())
        # Update brighteness sliders
        self.ui.slSat1low.setValue(minmaxes[0][0])
        self.ui.slSat1hi.setValue(minmaxes[0][1])
        self.ui.slSat2low.setValue(minmaxes[1][0])
        self.ui.slSat2hi.setValue(minmaxes[1][1])
        self.ui.slSat3low.setValue(minmaxes[2][0])
        self.ui.slSat3hi.setValue(minmaxes[2][1])
        # Update images
        images[0].updateImage(image=imgOut[0], levels=minmaxes[0])
        images[1].updateImage(image=imgOut[1], levels=minmaxes[1])
        images[2].updateImage(image=imgOut[2], levels=minmaxes[2])

    def allGCSText(self):
        # Pull GCS values
        CMElon = float(self.ui.leLon.text())
        CMElat = float(self.ui.leLat.text())
        CMEtilt = float(self.ui.leTilt.text())
        height = float(self.ui.leHeight.text())
        ang = float(self.ui.leAW.text())
        k = float(self.ui.leK.text())
        # Reset Sliders
        self.ui.sliderLon.setValue(CMElon)
        self.ui.sliderLat.setValue(CMElat)
        self.ui.sliderTilt.setValue(CMEtilt)
        self.ui.sliderHeight.setValue(int(height*10))
        self.ui.sliderAW.setValue(ang)
        self.ui.sliderK.setValue(int(k*100))
        # Make new wirerframes and plot
        data = getGCS(CMElon, CMElat, CMEtilt, height, k, ang, satpos, nleg=ns[0], ncirc=ns[1], ncross=ns[2])
        for i in range(nSats):
            self.plotGCSscatter(scatters[i], data[i], scaleNshift[i], innerR[i])


# Simple code to set up and run the GUI -------------------------------------------------|
def runGCSgui(ims, satpos, plotranges, sats, ns):
    # Make an application
    app = QtWidgets.QApplication([])
    # Make a widget
    application = mywindow(ims, satpos, plotranges, sats, nsIn=ns)
    # Run it
    application.show()
    # Exit nicely
    sys.exit(app.exec())
