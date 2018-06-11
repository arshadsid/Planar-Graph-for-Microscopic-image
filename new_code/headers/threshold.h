/***********************************************************************************************************************
 * Header file defines the thresholding methods to be used in program
***********************************************************************************************************************/
// C++ include files
#include <map>
using namespace std;

// OpenCV include files
#include <opencv2/core/core.hpp>
using namespace cv;

#ifndef __THRESHOLD_H_INCLUDED__
#define __THRESHOLD_H_INCLUDED__

multimap< int, Point > Threshold (Mat&, int);

#endif