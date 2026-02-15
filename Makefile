CC      = cc
CXX     = g++
CFLAGS  = -O3 -march=native -I include
CXXFLAGS= -O3 -march=native -std=c++17 -Wall -Wextra -I include
LDFLAGS = -lm

OBJDIR  = dist
TARGET  = $(OBJDIR)/dither

OBJS    = $(OBJDIR)/gifenc.o $(OBJDIR)/main.o

all: $(TARGET)

$(TARGET): $(OBJS) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/gifenc.o: source/gifenc.c include/gifenc.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/main.o: source/main.cpp include/stb_image.h include/gifenc.h | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR)

.PHONY: all clean
