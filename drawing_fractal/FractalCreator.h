#ifndef FRACTALCREATOR_H_
#define FRACTALCREATOR_H_

#include <string>
#include "Zoom.h"
#include <vector>
#include "Bitmap.h"
#include "Mandelbrot.h"
#include "Zoom.h"
#include "ZoomList.h"
#include "RGB.h"
#include <cstdint>
#include <memory>
#include <math.h>


using namespace std;

namespace caveofprogramming{

class FractalCreator{
private:	
	int m_width;
	int m_height;
	unique_ptr<int[]> m_histogram; 
	unique_ptr<int[]> m_fractal; 
	Bitmap m_bitmap;
	ZoomList m_zoomList;
	int m_total{0};

	vector<int> m_ranges;
	vector<RGB> m_colors;
	vector<int> m_rangeTotals;

	bool m_bGotFirstRange{false};
	
private:

	void calculateIteration();
	void calculateTotalIterations();
	void drawFractal();
	void writeBitmap(string name);
	void calculateRangeTotals();
	int getRange(int iterations) const;

public:
	FractalCreator(int width, int height);
	virtual ~FractalCreator();
	void addZoom(const Zoom &zoom);
	void addRange(double rangeEnd, const RGB &rgb);
	void run(string name);
};


}

#endif