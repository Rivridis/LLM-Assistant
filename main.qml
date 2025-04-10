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
    title: "AI Chat App"
    color: "#1e1e2f"
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
                        color: "#444"
                    }
                    contentItem: Image {
                        source: "qrc:/icons/settings.svg"
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
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: ["Chat", "Code", "Translate", "Summarize"]
                    delegate: Rectangle {
                        width: parent.width
                        height: 40
                        color: ListView.isCurrentItem ? "#3a3a6f" : "transparent"

                        Text {
                            anchors.centerIn: parent
                            color: "#fff"
                            text: modelData
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: ListView.view.currentIndex = index
                        }
                    }
                }
            }
        }

        // Chat Panel
        Rectangle {
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
                    text: "AI Assistant Chat"
                    color: "white"
                    font.pointSize: 18
                    font.bold: true
                    Layout.alignment: Qt.AlignHCenter
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
                        Layout.preferredHeight: 90
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
                        text: "Send"
                        Layout.preferredHeight: 50
                        background: Rectangle {
                            color: "#5566ff"
                            radius: 10
                        }

                        onClicked: {
                            if (mytext.text !== "") {
                                let userInput = mytext.text
                                chatModel.append({ sender: "User", message: userInput })
                                mytext.text = ""
                                msgfield.positionViewAtEnd()

                                // Let UI update, then block
                                Qt.callLater(() => {
                                    let val = backend.process(userInput)  // this can block
                                    chatModel.append({ sender: "AI", message: val })
                                    msgfield.positionViewAtEnd()
                                })
                            }
                        }         
                    }
                }
            }
        }
    }
}
//qmllint disable