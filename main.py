import sys
from pathlib import Path
from PySide6.QtCore import QObject, Slot, Signal, QThread, QMetaObject, Qt, Q_ARG
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtQml import QQmlApplicationEngine, QmlElement
from PySide6.QtQuickControls2 import QQuickStyle
import model, code_mode
from pdf_mode import PDFChatAssistant
import os
assistant = PDFChatAssistant()
os.environ["OPENTELEMETRY_PYTHON_CONTEXT"] = "contextvars"

QML_IMPORT_NAME = "mymodule"
QML_IMPORT_MAJOR_VERSION = 1
class Worker(QObject):
    finished = Signal()
    resultReady = Signal(str)

    @Slot(str)
    def process(self, text):
        result = model.process_chat(text)
        self.resultReady.emit(result)
        self.finished.emit()

    @Slot(str, str)
    def code(self, text, code):
        print(text)
        print("me")
        result = code_mode.process_chat(text, code)
        self.resultReady.emit(result)
        self.finished.emit()

    @Slot(str, str)
    def pdf(self, text, url):
        print(text)
        result = assistant.process_chat(text, url)
        self.resultReady.emit(result)
        self.finished.emit()

@QmlElement
class Backend(QObject):
    resultReady = Signal(str)

    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.worker = Worker()

        self.worker.moveToThread(self.thread)
        self.worker.resultReady.connect(self.handle_result)
        self.thread.start()

    @Slot(str)
    def process(self, text):
        QMetaObject.invokeMethod(
            self.worker,
            "process",
            Qt.QueuedConnection,
            Q_ARG(str, text)
        )
    
    @Slot(str, str)
    def code(self, text, code):
        QMetaObject.invokeMethod(
            self.worker,
            "code",
            Qt.QueuedConnection,
            Q_ARG(str, text),
            Q_ARG(str, code)
        )

    @Slot(str, str)
    def pdf(self, text, url):
        QMetaObject.invokeMethod(
            self.worker,
            "pdf",
            Qt.QueuedConnection,
            Q_ARG(str, text),
            Q_ARG(str, url)
        )

    @Slot(str)
    def handle_result(self, output):
        print(f"[Backend] Result: {output}")
        self.resultReady.emit(output)


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    QQuickStyle.setStyle("FluentWinUI3")
    engine = QQmlApplicationEngine()
    app_icon = QIcon("app_icon.ico")
    QGuiApplication.setWindowIcon(app_icon)
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())

