from PyQt5 import QtWidgets
import UI
import sys

class Window(QtWidgets.QWidget, UI.Ui_Form):
 
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())