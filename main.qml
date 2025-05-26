// qmllint disable
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Dialogs
import mymodule



ApplicationWindow {
    id: window
    visible: true
    width: 1000
    height: 600
    title: "Windows Assistant"
    color: "#1e1e2f"
    property string selectedMode: "Chat"
    Backend {
        id: backend
    }
    
    property string selectedFilePath: ""

    // tiny setting popup
    Popup {
        id: settingsPopup
        width: 500
        height: 400
        modal: true
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent
        anchors.centerIn: parent
        background: Rectangle {
            color: "transparent"
            border.width: 0
        }

        Rectangle {
            anchors.fill: parent
            color: "#2e2e4f"
            radius: 10

            ColumnLayout {
            anchors.fill: parent
            anchors.margins: 20
            spacing: 20

            Label {
                text: "Settings"
                color: "#bbb"
                font.pointSize: 18
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                Layout.alignment: Qt.AlignHCenter
            }

            Item { Layout.fillHeight: true }

            Button {
                text: "Download (coming soon)"
                Layout.alignment: Qt.AlignHCenter
                enabled: false
                opacity: 0.6
            }

            Item { Layout.fillHeight: true }

            Button {
                text: "Close"
                Layout.alignment: Qt.AlignHCenter
                onClicked: settingsPopup.close()
            }
            }
        }
    }


    FileDialog {
        id: file
        title: "Please choose a file"
        nameFilters: ["PDF files (*.pdf)"]
        onAccepted: {
            selectedFilePath = file.selectedFile
            console.log("You chose: " + selectedFilePath)
        }
        onRejected: {
            console.log("Canceled")
        }
    }

    Rectangle {
        anchors.fill: parent
        color: "#1e1e2f"

        // Sidebar
        Rectangle {
            width: 80
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            color: "#2e2e4f"

            Column {
                anchors.fill: parent
                spacing: 20
                padding: 20
                Item { Layout.fillHeight: true }

                // Settings Icon
                Button {
                    width: 40
                    height: 40
                    anchors.horizontalCenter: parent.horizontalCenter
                    background: Rectangle {
                        radius: 10
                        color: "#3a3a6f"
                    }
                    contentItem: Image {
                        source: "icons/settings.svg"
                        anchors.centerIn: parent
                        fillMode: Image.PreserveAspectFit
                    }

                    onClicked: {
                        settingsPopup.open()
                    }
                }
            }
        }

        // Mode Selection Panel
        Rectangle {
            width: 200
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.leftMargin: 80
            color: "#25253f"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 10

                Label {
                    text: "Modes"
                    color: "#bbb"
                    font.bold: true
                    font.pointSize: 14
                    horizontalAlignment: Text.AlignHCenter
                    Layout.alignment: Qt.AlignHCenter
                }

                ListView {
                    id: listViewRef
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: ["Chat", "Code", "PDF"]
                    delegate: Rectangle {
                        width: parent.width
                        height: 40
                        radius: 8
                        color: ListView.isCurrentItem ? "#3a3a6f" : "transparent"

                        Text {
                            anchors.centerIn: parent
                            color: "#fff"
                            text: modelData
                            font.pixelSize: 16
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: {
                                listViewRef.currentIndex = index
                                selectedMode = modelData
                            }
                        }
                    }
                }
            }
        }

        // Chat Panel
        Rectangle {
            visible: selectedMode === "Chat"
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.leftMargin: 280
            anchors.right: parent.right
            color: "#1f1f2f"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 10

                Label {
                    text: "Vivy - AI Assistant"
                    color: "#bbb"
                    font.pointSize: 18
                    font.bold: true
                    Layout.alignment: Qt.AlignHCenter
                    padding : 10
                }

                ListModel {
                    id: chatModel
                    ListElement { sender: "AI"; message: "Hello, This is Vivy, how can I help you today?" }
                }

                // Clear chatModel when switching from Code to Chat mode
                Connections {
                    target: window
                    function onSelectedModeChanged() {
                        if (selectedMode === "Chat") {
                            chatModel.clear()
                            chatModel.append({ sender: "AI", message: "Hello, This is Vivy, how can I help you today?" })
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#2c2c3f"
                    radius: 10

                    ListView {
                        id: msgfield
                        anchors.fill: parent
                        model: chatModel
                        clip: true

                        delegate: Column {
                            width: ListView.view.width
                            spacing: 4
                            padding: 10

                            Text {
                                text: model.sender + ": " + model.message
                                color: model.sender === "User" ? "#fff" : "#bbb"
                                wrapMode: Text.Wrap
                                width: parent.width - 40
                                font.pointSize: 14
                            }
                        }
                    }
                }

                // Input Area
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Layout.preferredHeight: 60

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 50
                        radius: 10
                        color: "#2c2c3f"
                        clip: true

                        ScrollView {
                            anchors.fill: parent
                            clip: true

                            TextArea {
                                id: mytext
                                wrapMode: TextArea.Wrap
                                placeholderText: "Type your message..."
                                color: "#fff"
                                font.pixelSize: 16
                                background: null  // remove default background
                                padding: 10
                            }
                        }
                    }
                

                    Button {
                        id: send
                        text: "Send"
                        Layout.preferredHeight: 47
                        background: Rectangle {
                            color: "#5566ff"
                            radius: 10
                        }
                        enabled: true  // Initially enabled

                        onClicked: {
                            if (mytext.text !== "") {
                                send.enabled = false
                                let userInput = mytext.text
                                chatModel.append({ sender: "User", message: userInput })
                                mytext.text = ""
                                msgfield.positionViewAtEnd()
                                backend.process(userInput)

                            }
                        }         
                    }
                Connections {
                target: backend
                function onResultReady(val){
                    chatModel.append({ sender: "AI", message: val })
                    send.enabled = true
                    msgfield.positionViewAtEnd()
                        }
                    }
                }
            }
        }

        // Code editor panel with line numbering and current line highlighting
        Rectangle {
            visible: selectedMode === "Code"
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.leftMargin: 280
            anchors.right: parent.right
            color: "#1f1f2f"

            ColumnLayout {
            anchors.fill: parent
            anchors.margins: 20
            spacing: 10

            Label {
                text: "Code Editor"
                color: "#bbb"
                font.pointSize: 18
                font.bold: true
                Layout.alignment: Qt.AlignHCenter
                padding: 10
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#2c2c3f"
                radius: 10

                RowLayout {
                anchors.fill: parent
                spacing: 0

                // Line numbers
                ListView {
                    id: lineNumberView
                    width: 40
                    Layout.fillHeight: true
                    model: codeArea.lineCount
                    interactive: false
                    clip: true
                    delegate: Rectangle {
                    width: ListView.view.width
                    height: codeArea.lineHeight
                    color: index === codeArea.cursorLine ? "#3a3a6f" : "transparent"
                    Text {
                        anchors.centerIn: parent
                        text: (index + 1).toString()
                        color: "#888"
                        font.pointSize: 14
                    }
                    }
                    // Sync vertical scroll with codeArea
                    property bool ignoreSync: false
                    onContentYChanged: {
                    if (!ignoreSync && !codeScroll.ignoreSync && Math.abs(contentY - codeFlickable.contentY) > 1) {
                        codeScroll.ignoreSync = true
                        codeFlickable.contentY = contentY
                        codeScroll.ignoreSync = false
                    }
                    }
                }

                ScrollView {
                    id: codeScroll
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    property bool ignoreSync: false
                    contentItem: Flickable {
                    id: codeFlickable
                    contentWidth: codeArea.width
                    contentHeight: codeArea.height
                    anchors.fill: parent
                    onContentYChanged: {
                        if (!codeScroll.ignoreSync && !lineNumberView.ignoreSync && Math.abs(contentY - lineNumberView.contentY) > 1) {
                        lineNumberView.ignoreSync = true
                        lineNumberView.contentY = contentY
                        lineNumberView.ignoreSync = false
                        }
                    }
                    TextArea {
                        id: codeArea
                        wrapMode: TextArea.Wrap
                        color: "#fff"
                        font.family: "monospace"
                        font.pixelSize: 16
                        background: null
                        selectByKeyboard: true
                        verticalAlignment: TextInput.AlignTop
                        padding: 0

                        property int lineCount: text.length === 0 ? 1 : text.split("\n").length
                        property int lineHeight: font.pixelSize + 4
                        property int cursorLine: {
                        let t = text.substring(0, codeArea.cursorPosition)
                        if (t.length === 0)
                            return 0
                        // If the cursor is at a newline character, highlight the previous line
                        if (codeArea.cursorPosition > 0 && text[codeArea.cursorPosition - 1] === "\n" && codeArea.cursorPosition === text.length)
                            return t.split("\n").length - 1
                        if (codeArea.cursorPosition > 0 && text[codeArea.cursorPosition - 1] === "\n")
                            return t.split("\n").length - 1
                        return t.split("\n").length - 1
                        }
                        topPadding: 0
                        bottomPadding: 0

                        property int lastCursorPosition: 0

                        function isCursorAtEnd() {
                        return codeArea.cursorPosition === codeArea.text.length
                        }

                        function isCursorAtLastLine() {
                        return codeArea.cursorLine === codeArea.lineCount - 1
                        }

                        function scrollToLastLine() {
                        Qt.callLater(function() {
                            let totalHeight = codeArea.lineCount * codeArea.lineHeight
                            let flickHeight = codeFlickable.height
                            if (totalHeight > flickHeight) {
                            codeFlickable.contentY = totalHeight - flickHeight
                            }
                        })
                        }

                        property real lastContentY: 0

                        onTextChanged: {
                        // Only scroll if the cursor is at the last line
                        if (isCursorAtLastLine()) {
                            scrollToLastLine()
                        } else {
                            var prevContentY = codeFlickable.contentY
                            Qt.callLater(function() {
                            codeFlickable.contentY = prevContentY
                            })
                        }
                        }
                    }
                    }
                }
                }
            }

            // Input Area below the code editor
            RowLayout {
                Layout.fillWidth: true
                spacing: 10
                Layout.preferredHeight: 60

                Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 50
                radius: 10
                color: "#2c2c3f"
                clip: true

                ScrollView {
                    anchors.fill: parent
                    clip: true

                    TextArea {
                    id: codeInput
                    wrapMode: TextArea.Wrap
                    placeholderText: "Type your code prompt..."
                    color: "#fff"
                    font.pixelSize: 16
                    background: null
                    padding: 10
                    }
                }
                }

                Button {
                id: codeSend
                text: "Send"
                Layout.preferredHeight: 47
                background: Rectangle {
                    color: "#5566ff"
                    radius: 10
                }
                enabled: true

                onClicked: {
                    if (codeInput.text !== "") {
                    codeSend.enabled = false
                    let userInput = codeInput.text
                    let code = codeArea.text
                    // You can handle the code input here, e.g., send to backend
                    backend.code(userInput, code)
                    codeInput.text = ""
                    }
                }
                }
                Connections {
                target: backend
                function onResultReady(val){
                    // You can handle the backend response here, e.g., insert into codeArea or show a message
                    codeSend.enabled = true
                    codeArea.text = ""
                    codeArea.text = val
                }
                }

                Connections {
                    target: window
                    function onSelectedModeChanged() {
                        if (selectedMode === "Code") {
                            codeArea.text = ""
                        }
                    }
                }
            }
            }
        }

        // PDF viewer panel
       Rectangle {
            visible: selectedMode === "PDF"
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.leftMargin: 280
            anchors.right: parent.right
            color: "#1f1f2f"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 10

                Label {
                    text: "PDF mode"
                    color: "#bbb"
                    font.pointSize: 18
                    font.bold: true
                    Layout.alignment: Qt.AlignHCenter
                    padding : 10
                }

                ListModel {
                    id: chatModelP
                    ListElement { sender: "AI"; message: "Please upload a PDF document to continue" }
                }

                // Clear chatModel when switching from Code to Chat mode
                Connections {
                    target: window
                    function onSelectedModeChanged() {
                        if (selectedMode === "PDF") {
                            chatModelP.clear()
                            chatModelP.append({ sender: "AI", message: "Please upload a PDF document to continue" })
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#2c2c3f"
                    radius: 10

                    ListView {
                        id: msgfieldP
                        anchors.fill: parent
                        model: chatModelP
                        clip: true

                        delegate: Column {
                            width: ListView.view.width
                            spacing: 4
                            padding: 10

                            Text {
                                text: model.sender + ": " + model.message
                                color: model.sender === "User" ? "#fff" : "#bbb"
                                wrapMode: Text.Wrap
                                width: parent.width - 40
                                font.pointSize: 14
                            }
                        }
                    }
                }

                // Input Area
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Layout.preferredHeight: 60

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 50
                        radius: 10
                        color: "#2c2c3f"
                        clip: true

                        ScrollView {
                            anchors.fill: parent
                            clip: true

                            TextArea {
                                id: mytextP
                                wrapMode: TextArea.Wrap
                                placeholderText: "Type your message..."
                                color: "#fff"
                                font.pixelSize: 16
                                background: null  // remove default background
                                padding: 10
                            }
                        }
                    }
                

                    Button {
                        id: sendP
                        text: "Send"
                        Layout.preferredHeight: 47
                        background: Rectangle {
                            color: "#5566ff"
                            radius: 10
                        }
                        enabled: true  // Initially enabled

                        onClicked: {
                            if (mytextP.text !== "" && selectedFilePath !== "") {
                                sendP.enabled = false
                                let userInput = mytextP.text
                                chatModelP.append({ sender: "User", message: userInput })
                                mytextP.text = ""
                                msgfieldP.positionViewAtEnd()
                                backend.pdf(userInput, selectedFilePath)

                            }
                        }         
                    }

                    Button {
                        id: upload
                        text: "Upload"
                        Layout.preferredHeight: 47
                        background: Rectangle {
                            color: "#5566ff"
                            radius: 10
                        }
                        enabled: true  // Initially enabled

                        onClicked: {
                            file.open()
                        }         
                    }

                Connections {
                target: backend
                function onResultReady(val){
                    chatModelP.append({ sender: "AI", message: val })
                    sendP.enabled = true
                    msgfieldP.positionViewAtEnd()
                        }
                    }
                }
            }
        }

    }
}


