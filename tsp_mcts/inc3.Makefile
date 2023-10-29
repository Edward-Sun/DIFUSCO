#******************************************************************************
#
#
# ******************************************************************************
#  File Name     : MAKEFILE
#  Version       : V1.0
#  Author        : shliu
#  Created       : 2017/08/19
#  Description   : makefile
#
#******************************************************************************/

MAKE = make
CD = cd

CROSS = ${_CROSS_COMPILER_}

CC := $(CROSS)gcc
CXX := $(CROSS)g++
LD := $(CROSS)ld
AR := $(CROSS)ar
AS := $(CROSS)as
STRIP := $(CROSS)strip
LINKTOOL = $(CROSS)g++

#�Ƿ�debugģ��
ifeq "$(IS_DEBUG)" "1"
	CFLAGS += -g
	SOCFLAGS += -g
endif

CFLAGS += -fpic  -Wall -O3

SOCFLAGS += -fpic  -shared

SOCPPFLAGS +=  -shared

SOCPPFLAGS += $(CPPFLAGS)

STACFLAGS = ar cr

#���ӿ�ִ���ļ���ʱ���õ�
LDFLAGS += -lpthread -lm -ldl

SRC := $(strip $(SRC))
OBJ := $(addsuffix .o, $(basename $(SRC)))

LIBS := $(addsuffix .a, $(STATIC_LIBS))

OBJ_C := $(patsubst %.c, %.o, $(C_SRC))
OBJ_CPP := $(patsubst %.cpp, %.o, $(CPP_SRC))


CPPFLAGS += -lstdc++ -std=c++11
#CPPFLAGS += -lstdc++
CPPFLAGS += -fPIC

%.o : %.c
	$(CC)  -o $@ -c $< $(CFLAGS) $(INCLUDE_DIR)

%.o : %.cpp
	$(CXX) $(CPPFLAGS) -fvisibility=hidden -o  $@ -c $<  $(INCLUDE_DIR)

#ZKLZFaceRecognizer.o : ZKLZFaceRecognizer.cpp
#	$(CXX) $(CPPFLAGS) -o ZKLZFaceRecognizer.o -c ZKLZFaceRecognizer.cpp $(INCLUDE_DIR)

#############################
## rule to make the target ##
#############################
.PHONY: all
all: subtarget $(TARGET)

subtarget:
	@echo "start make sub directory makefile..."
	@for i in $(SUBDIR); do \
		@echo $$i; \
		$(MAKE) -C $$i; \
	done

#��̬�����
ifeq "$(TARGET_TYPE)" "SHARED_LIBRARY"
$(TARGET):$(OBJ)  $(OBJ_C) $(OBJ_CPP)
	$(LINKTOOL) $(SOCPPFLAGS) -o $@ $^  $(DEP_LIB_DIR) $(LIBS) $(OPENCV_LIBS)  ${OTHER_LIBS} $(LDFLAGS)

ifeq "$(IS_DEBUG)" "0"
	$(STRIP) $@
endif
	#cp -fp $(TARGET) $(COPY_TO)
	#cp -fp $(TARGET) ../FaceTest/
	#cp -fp ./ZKLZFaceRecognizer.h ../FaceTest/
	#cp -rf $(MAP_INC) $(APP_INC)
	rm -rf *.o
	#rm -rf ./ZKLZTools/*.o
	#rm -rf ./License/*.o
	@echo "CREAT *$(TARGET)* SUCCESS!!!!!!!!!!!!!!!!!"
	@echo
endif

#��̬�����
ifeq "$(TARGET_TYPE)" "STATIC_LIBRARY"
$(TARGET):$(OBJ)  $(OBJ_C) $(OBJ_CPP)
	$(STACFLAGS) -o $@ $^ $(LIBS)
	cp -fp $(TARGET) $(COPY_TO)
	cp -rf $(MAP_INC) $(APP_INC)
	rm -rf *.o
	@echo "CREAT *$(TARGET)* SUCCESS!!!!!!!!!!!!!!!!!"
	@echo
endif

#������Ƭ��ִ�г���$(LIBS)
ifeq "$(TARGET_TYPE)" "EXE_FILE"
$(TARGET): $(OBJ)  $(OBJ_C) $(OBJ_CPP)
	$(LINKTOOL) $(CPPFLAGS) -o $@ $^  $(DEP_LIB_DIR) $(LIBS) $(OPENCV_LIBS)   $(OPENCV_3RD_LIBS) ${OTHER_LIBS} $(LDFLAGS)
ifeq "$(IS_DEBUG)" "0"
	$(STRIP) $@
endif
	chmod a+x $@
	rm -rf *.o
#	cp -f $(TARGET) $(COPY_TO)
#	cp -rf $(MAP_INC) $(APP_INC)
	@echo "make exe succesfull!!!"
endif

$(OBJ):%o:%c
	$(CXX) $(CFLAGS) -c $< -o $@  $(INCLUDE_DIR)


.PHONY: clean
clean:
	rm $(OBJ) $(TARGET) \
	@for i in $(SUBDIR); do \
		$(MAKE) clean -C $$i; \
	done
