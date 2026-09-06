CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall
TARGET   := aruco_navigation
SRC      := src/main.cpp
PKG      := $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv 2>/dev/null)

.PHONY: all clean run markers

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(PKG)

markers: $(TARGET)
	./$(TARGET) --generate-markers

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) frame_*.png
	rm -rf markers evaluation
