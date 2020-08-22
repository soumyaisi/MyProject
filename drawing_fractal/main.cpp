#include <iostream>
#include <cstdint>
#include <memory>
#include <math.h>
#include "FractalCreator.h"
#include "RGB.h"
#include "Zoom.h"

using namespace std;
using namespace caveofprogramming;

int main(){	

	//int height = 600;
	FractalCreator fractalCreator(800, 600);

	fractalCreator.addRange(0.0, RGB(0,0,0));
	fractalCreator.addRange(0.3, RGB(255,0,0));
	fractalCreator.addRange(0.5, RGB(255,255,0));
	fractalCreator.addRange(1.0, RGB(255,255,255));

	//fractalCreator.getRange(200)

	//fractalCreator.addZoom(Zoom(295, height -202, 0.1));
	//fractalCreator.addZoom(Zoom(312, height -304, 0.1));
	//fractalCreator.calculateIteration();
	//fractalCreator.calculateTotalIterations();
	//fractalCreator.drawFractal();
	//fractalCreator.writeBitmap("test6.bnp");


	fractalCreator.addZoom(Zoom(295,  202, 0.1));
	fractalCreator.addZoom(Zoom(312,  304, 0.1));
	fractalCreator.run("test6.bnp");

	cout << "finished" << endl;
	return 0;
}

// To run...... g++ -std=c++11 main.cpp Bitmap.cpp Mandelbrot.cpp ZoomList.cpp FractalCreator.cpp RGB.cpp
// Check Memory leak.....use valgrind;