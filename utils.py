import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QComboBox, QCompleter

def polygon2bbox(pts):
    x = np.min(pts[:, 0])
    y = np.min(pts[:, 1])
    w = np.max(pts[:, 0]) - x
    h = np.max(pts[:, 1]) - y
    return [int(x), int(y), int(w), int(h)]

def bbox2polygon(box):
    x, y, w, h = box
    # starting with left bottom point then anti clockwise
    return [x, y + h, x + w, y + h, x + w, y, x, y]

def completion(word_list, widget, i=True):
    """ Autocompletion of sender and subject """
    word_set = set(word_list)
    completer = QCompleter(word_set)
    if i:
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
    else:
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitive)
    completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
    widget.setCompleter(completer)


class Autocomplete(QComboBox):
    def __init__(self, items, parent=None, i=False, allow_duplicates=True):
        super(Autocomplete, self).__init__(parent)
        self.items = items
        self.insensitivity = i
        self.allowDuplicates = allow_duplicates
        self.init()

    def init(self):
        self.setEditable(True)
        self.setDuplicatesEnabled(self.allowDuplicates)
        self.addItems(self.items)
        self.setAutocompletion(self.items, i=self.insensitivity)

    def setAutocompletion(self, items, i):
        completion(items, self, i)


