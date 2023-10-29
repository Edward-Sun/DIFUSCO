#******************************************************************************
#
#
# ******************************************************************************
#  File Name     : MAKEFILE
#  Version       : V1.0
#  Description   : makefile
#
#******************************************************************************/

#工程目录
#PROJECT_DIR=/home/dell/桌面/test/

#拷贝目录

#Lib
export DEP_LIB_DIR = -L.
#交叉编译工具
_CROSS_COMPILER_=

#公共库
#_COMMON_DIR=$(OPENCV_DIR)lib/

#静态库
#export LIBS += $(OPENCV_DIR)

#是否调试
export IS_DEBUG = 0


#目标文件 0表示可执行文件，1表示动态库，2表示静态库
ObjType =0
ifeq "$(ObjType)" "0"
	TARGET_TYPE := EXE_FILE
	TARGET      := test
else ifeq "$(ObjType)" "1"
	TARGET_TYPE := SHARED_LIBRARY
	TARGET      := libFace.so
else ifeq "$(ObjType)" "2"
	TARGET_TYPE := STATIC_LIBRARY
	TARGET      := libFace.a
endif

#头文件
#INCLUDE_DIR= -I$(OPENCV_DIR)include/ -I${OPENBLAS_INC} -I${NCNN_INC} -I${NCNN_BUILD_INC}

#动态库
#DYNAMIC_LINK_DIR = -lopencv_core -lopencv_video -lopencv_objdetect -lopencv_imgproc -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_ml #-lopencv_contrib -lopencv_legacy -lopencv_nonfree -lopencv_ts

#OPENCV_LIBS = libopencv_calib3d libopencv_imgcodecs libopencv_imgproc libopencv_video libopencv_videoio libopencv_flann libopencv_core libopencv_highgui
#OPENCV_LIBS = -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_videoio -lopencv_flann -lopencv_highgui   -lopencv_photo -lopencv_shape  -lopencv_ml -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_core -lopenblas -lncnn


#ifeq "$(ObjType)" "0"
#OPENCV_LIBS += -lgomp
#endif

#OPENCV_3RD_LIBS= -llibtiff -llibpng -llibjpeg -llibjasper -lIlmImf -llibwebp -lzlib
#OTHER_LIBS =  -L. libZKLZTools.a
#STATIC_LIBS =  ${OPENCV_LIBS} ${OPENCV_3RD_LIBS} -lZKLZTools
STATIC_LIBS_DIR = -L. ${_COMMON_DIR} -L${NCNN_LIB}  -L${OPENBLAS_LIB}
#STATIC_LIBS_DIR =  -L. ${OPENCV_3RD_DIR} -L. -L${NCNN_LIB} -L${OPENBLAS_LIB}

#源码目录
CPP_SRC := $(shell find ./ -name "*.cpp")



include ./inc3.Makefile
