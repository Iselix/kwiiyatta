import copy

from PySide2 import QtCore, QtWidgets

import kwiiyatta

from .ui import kwiieiya


class KwiieiyaDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, conf=None):
        super().__init__(
            parent,
            (
                QtCore.Qt.WindowSystemMenuHint
                | QtCore.Qt.WindowTitleHint
                | QtCore.Qt.WindowCloseButtonHint
            )
        )
        self.ui = kwiieiya.Ui_KwiieiyaDialog()
        self.ui.setupUi(self)

        self.ui.openSourceButton.clicked.connect(self.openSource)
        self.ui.playSourceButton.clicked.connect(self.playSource)
        self.ui.openCarrierButton.clicked.connect(self.openCarrier)
        self.ui.playCarrierButton.clicked.connect(self.playCarrier)
        self.ui.saveButton.clicked.connect(self.save)
        self.ui.playButton.clicked.connect(self.play)

        if conf is None:
            conf = kwiiyatta.Config()
            conf.parse_args()

        self.conf = conf

    def openFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, filter='音声ファイル (*.wav)'
        )
        if filename == '':
            return None, None

        wav = kwiiyatta.load_wav(filename)
        if len(wav.data.shape) == 2:
            QtWidgets.QMessageBox.information(
                self, 'kwiieiya',
                ('指定したファイルはモノラルではありません\n'
                 'チャンネル 1 を読み込みます')
            )
            wav.data = wav.data[:, 0]
        return filename, self.conf.create_analyzer(wav)

    def extract_feature(self, feature, ape_threshold, ape_threshold_spinbox):
        new_ape_threshold = ape_threshold_spinbox.value()
        if feature is not None:
            if ape_threshold != new_ape_threshold:
                feature.clear_features()
                feature.extract_aperiodicity(threshold=new_ape_threshold)
        return feature, new_ape_threshold

    def saveFile(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, filter='音声ファイル (*.wav)'
        )
        return filename if filename != '' else None

    def extractSource(self):
        self.source, self.sourceApeThreshold = \
            self.extract_feature(
                self.source,
                self.sourceApeThreshold, self.ui.sourceApeThresholdSpinBox)

    def openSource(self):
        filename, source = self.openFile()
        if source is not None:
            self.source = source
            self.sourceApeThreshold = None
            self.ui.sourceFilename.setText(str(filename))
            self.ui.sourceFs.setText(str(source.fs))
            self.ui.playSourceButton.setEnabled(True)
            self.ui.playButton.setEnabled(True)
            self.ui.saveButton.setEnabled(True)

    def playSource(self):
        self.extractSource()
        self.source.synthesize().play()

    def extractCarrier(self):
        self.carrier, self.carrierApeThreshold = \
            self.extract_feature(
                self.carrier,
                self.carrierApeThreshold, self.ui.carrierApeThresholdSpinBox)

    def openCarrier(self):
        filename, carrier = self.openFile()
        if carrier is not None:
            self.carrier = carrier
            self.carrierApeThreshold = None
            self.ui.carrierFilename.setText(str(filename))
            self.ui.carrierFs.setText(str(carrier.fs))
            self.ui.playCarrierButton.setEnabled(True)

    def playCarrier(self):
        self.extractCarrier()
        self.carrier.synthesize().play()

    def synthesize(self):
        self.extractSource()
        feature = kwiiyatta.feature(self.source)

        if self.ui.carrierBox.isChecked():
            self.extractCarrier()
            feature = kwiiyatta.align(self.source, self.carrier)
            if self.ui.useDiffvcCheckBox.isChecked():
                mcep_diff = copy.copy(feature.mel_cepstrum)
                mcep_diff.data -= self.carrier.mel_cepstrum.data
                return kwiiyatta.apply_mlsa_filter(self.carrier.wavdata,
                                                   mcep_diff)
            else:
                feature.f0 = self.carrier.f0

        if self.ui.useMcepCheckBox.isChecked():
            feature.extract_mel_cepstrum()
            feature.spectrum_envelope = None
        return feature.synthesize()

    def save(self):
        filename = self.saveFile()
        if filename is not None:
            self.synthesize().save(filename)

    def play(self):
        self.synthesize().play()
