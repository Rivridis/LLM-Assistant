// qmllint disable
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import mymodule

ApplicationWindow {
    id: window
    visible: true
    width: 1000
    height: 600
    title: "Vivy"
    color: "#1e1e2f"
    property string selectedMode: "Chat"
    Backend {
        id: backend
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
                }

                ListView {
                    id: listViewRef
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: ["Chat", "Code"]
                    delegate: Rectangle {
                        width: parent.width
                        height: 40
                        radius: 8
                        color: ListView.isCurrentItem ? "#3a3a6f" : "transparent"

                        Text {
                            anchors.centerIn: parent
                            color: "#fff"
                            text: modelData
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: listViewRef.currentIndex = index
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
                            width: parent.width
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
    }
}
//qmllint disable