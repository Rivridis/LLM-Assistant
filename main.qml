import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

ApplicationWindow {
    id: window
    visible: true
    width: 1000
    height: 600
    title: "AI Chat App"
    color: "#1e1e2f"

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

                Item { Layout.fillHeight: true } // Push settings to the bottom

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

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#2c2c3f"
                    radius: 10

                    // Simulated chat content
                    Flickable {
                        anchors.fill: parent
                        contentHeight: columnContent.height
                        clip: true

                        Column {
                            id: columnContent
                            width: parent.width
                            spacing: 12
                            padding: 10

                            Text {
                                text: "User: Hello AI!"
                                color: "#fff"
                                wrapMode: Text.Wrap
                                width: parent.width - 40
                            }

                            Text {
                                text: "AI: Hello, how can I help you today?"
                                color: "#bbb"
                                wrapMode: Text.Wrap
                                width: parent.width - 40
                            }

                            Text {
                                text: "User: Can you summarize this article?"
                                color: "#fff"
                                wrapMode: Text.Wrap
                                width: parent.width - 40
                            }

                            Text {
                                text: "AI: Sure! Please paste the article here."
                                color: "#bbb"
                                wrapMode: Text.Wrap
                                width: parent.width - 40
                            }
                        }
                    }
                }

                // Input Area
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Layout.preferredHeight: 60

                    TextField {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 50
                        placeholderText: "Type your message..."
                        color: "#fff"
                        font.pixelSize: 16
                        background: Rectangle {
                            color: "#333"
                            radius: 10
                        }
                    }

                    Button {
                        text: "Send"
                        Layout.preferredHeight: 50
                        background: Rectangle {
                            color: "#5566ff"
                            radius: 10
                        }
                    }
                }
            }
        }
    }
}
