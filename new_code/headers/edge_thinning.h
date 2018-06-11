/***********************************************************************************************************************
 * Header file defines the edge thinning methods to be used in program
***********************************************************************************************************************/
// C++ include files
#include <iostream>
#include <string>
#include <set>
#include <map>

// OpenCV include files
#include <opencv2/core/core.hpp>
using namespace cv;

#ifndef __EDGE_THINNING_H_INCLUDED__
#define __EDGE_THINNING_H_INCLUDED__

struct Compare_Points
{
    inline bool operator() (cv::Point const& a, cv::Point const& b) const
    {
        if ( a.x < b.x ) return true;
        if ( a.x == b.x and a.y < b.y ) return true;
        return false;
    }
};

extern std::vector<Point> shifts8;
extern std::vector<Point> shifts4;

bool Edge_Thinning (Mat const&, Mat_<bool>&, std::vector<int>&, bool&, std::string);
bool Save_Mask (Mat const&, Mat const&, std::string);
bool Remove_External_Pixels (Mat_<bool>&);
bool Add_Internal_Pixels (Mat_<bool>&);
bool Neighbors (Mat_<bool>const&, Point, std::vector<Point>const&, int&, std::vector<bool>&);
bool Remove_Small_Components (bool, int, Mat_<bool>&, int&);

#endif