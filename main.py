import sys
from pathlib import Path
from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle

QML_IMPORT_NAME = "mymodule"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class Backend(QObject):
    @Slot(str, result=str)
    def process(self, text):
        print(f"QML sent: {text}")
        # Replace this with your actual AI/LLM processing
        return f"AI Response: You said '{text}'"



if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    QQuickStyle.setStyle("FluentWinUI3")
    engine = QQmlApplicationEngine()
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())

