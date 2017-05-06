all: test
test: test.cpp
	g++ test.cpp -lopencv_core -lopencv_highgui -lopencv_legacy -lopencv_nonfree -lopencv_features2d -lopencv_calib3d -o test -g -O2
