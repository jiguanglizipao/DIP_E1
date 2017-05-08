all: ex3
ex3: ex3.cpp
	g++ ex3.cpp -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_legacy -lopencv_nonfree -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -o ex3 -g -O2
clean:
	rm -f ex3
