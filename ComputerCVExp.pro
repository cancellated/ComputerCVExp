QT += core gui widgets
CONFIG += c++17

# OpenCV配置
INCLUDEPATH += $$PWD/thirdparty/opencv/include

# OpenCV库文件路径
win32 {
    CONFIG(debug, debug|release) {
        LIBS += -L$$PWD/thirdparty/opencv/vc16/lib \
                -lopencv_world4110d
    } else {
        LIBS += -L$$PWD/thirdparty/opencv/vc16/lib \
                -lopencv_world4110
    }
}

# 源代码和界面文件
SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS +=\
    mainwindow.h

FORMS +=\
    mainwindow.ui

# 平台架构指定
win32 {
    QMAKE_LFLAGS += /MACHINE:X64
}


