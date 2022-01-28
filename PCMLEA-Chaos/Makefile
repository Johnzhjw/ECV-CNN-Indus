
VERSION = 1.0.0

CC = mpicxx
LD = mpicxx
SRCDIR = src
BUILDDIR = build
TARGET = DPNeuEvo

    #library file name
#LIBS = -ldylib
    #library file path
#LIB_PATH = -L./lib

    #macro definition
#DEBUG = -D_MACRO

SRCEXT = cpp
SOURCES = $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS = $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
DEP = $(OBJECTS:%.o=%.d)
CFLAGS = -Wall -c -pedantic -Ofast -std=c++0x
LDFLAGS = -mkl
INC = -I $(SRCDIR)

$(TARGET):$(OBJECTS)
	@echo "Linking..."
	$(LD) -o $(TARGET).$(VERSION)  $^ $(LDFLAGS) 

-include $(DEP)
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(sort $(dir $(OBJECTS)))
	$(CC) $(CFLAGS) $(INC) $(DEBUG) -MM -MT $@ -MF $(patsubst %.o, %.d, $@) $<
	$(CC) $(CFLAGS) $(INC) $(DEBUG) -c -o $@ $<

#$(CC) $(CFLAGS)  -MMD -c -o $@ $<
clean:
	@echo "cleaning...";
	$(RM) -r $(BUILDDIR) $(TARGET).$(VERSION)
.PHONY:clean
