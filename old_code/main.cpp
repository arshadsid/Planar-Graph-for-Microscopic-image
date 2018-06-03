#include <fstream>
#include <iostream>
#include <deque>
#include <set>
#include <string>
#include <iterator>
#include <utility>
#include <algorithm>
#include <limits>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

//Boost
#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/graph/graphviz.hpp>
#include "boost/graph/topological_sort.hpp"
#include <boost/graph/graph_traits.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/connected_components.hpp>
typedef boost::adjacency_list< boost::listS, boost::vecS, boost::undirectedS, Point > Graph;

// Constants
const bool debug = false;
const Point2d origin(0, 0);
const double big_constant = 1e+32;
const Vec3b black( 0, 0, 0 );
const Vec3b white( 255, 255, 255 );
const Scalar White = CV_RGB (255, 255, 255);
const Scalar Black = CV_RGB (0, 0, 0);
const Scalar Red = CV_RGB (255, 0, 0);
const Scalar Lime = CV_RGB (0, 255, 0);
const Scalar Green = CV_RGB (0, 128, 0);
const Scalar Blue = CV_RGB (0, 0, 255);
const Scalar Yellow = CV_RGB (255, 255, 0);
const Scalar Orange = CV_RGB (255, 128, 0);
const Scalar Magenta = CV_RGB (255, 0, 255);
const Scalar Cyan = CV_RGB (0, 255, 255);
const Scalar Olive = CV_RGB (128,128,0);
const Scalar Gray = CV_RGB (128,128,128);
const Scalar Silver = CV_RGB (192,192,192);
const Scalar Maroon = CV_RGB (128,0,0);
const Scalar Purple = CV_RGB (128,0,128);
const Scalar Teal = CV_RGB (0,128,128);
const Scalar Navy = CV_RGB (0,0,128);
const Scalar Brown = CV_RGB (165,42,42);
const Scalar Coral = CV_RGB (255,127,80);
const Scalar Salmon = CV_RGB (250,128,114);
const Scalar Khaki = CV_RGB (240,230,140);
const Scalar Indigo = CV_RGB (75,0,130);
const Scalar Plum = CV_RGB  (221,160,221);
const Scalar orchid = CV_RGB (218,112,214);
const Scalar beige = CV_RGB (245,245,220);
const Scalar peru = CV_RGB (205,133,63);
const Scalar sienna = CV_RGB (160,82,45);
const Scalar lavender = CV_RGB (230,230,250);
const Scalar mocassin = CV_RGB (255,228,181);
const Scalar honeydew = CV_RGB (240,255,240);
const Scalar ivory = CV_RGB (255,255,240);
const Scalar azure = CV_RGB (240,255,255);
const Scalar crimson = CV_RGB (220,20,60);
const Scalar gold = CV_RGB (255,215,0);
const Scalar sky_Blue = CV_RGB (135,206,235);
const Scalar aqua_marine = CV_RGB (127,255,212);
const Scalar bisque = CV_RGB (255,228,196);
const Scalar peach_puff = CV_RGB (255,218,185);
const Scalar corn_silk = CV_RGB (255,248,220);
const Scalar wheat = CV_RGB (245,222,179);
const Scalar violet = CV_RGB (238,130,238);
const Scalar lawn_green  = CV_RGB (124,252,0);
std::vector<CvScalar> BGR = { Blue, Green, Red };
std::vector<CvScalar> colors = { Lime, Orange, Gray, Cyan, Yellow, Red, Green, Olive, Magenta, Teal, Silver, Coral, Salmon, Khaki, Plum, orchid, beige, lavender, mocassin, honeydew, ivory, azure, crimson, gold, sky_Blue, aqua_marine, bisque, peach_puff, corn_silk, wheat, violet, lawn_green};
// dark colors: Blue, Black, Brown, Maroon, Navy, Purple, Indigo, peru, sienna,
const std::vector<Vec3b> Colormap_jet{ // from bright red to light blue through green, yellow
    Vec3b(139, 0, 0),
    Vec3b(0, 0, 0),
    Vec3b(0, 69, 139),
    Vec3b(0, 0, 128),
    Vec3b(128, 0, 0),
    Vec3b(128, 0, 128),
    Vec3b(130, 0, 75),
    Vec3b(63, 133, 205),
    Vec3b(45, 82, 160),
    Vec3b(0, 0, 139) };
std::vector<Point> shifts4 = { Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
std::vector<Point> shifts8 = { Point(1,0), Point(1,1), Point(0,1), Point(-1,1), Point(-1,0), Point(-1,-1), Point(0,-1), Point(1,-1) };

class Pixel
{
public:
    Point point;
    int value;
    Pixel (Point p, int v) { point = p; value = v; }
    void Print() { std::cout<<" v"<<point<<"="<<value<< std::endl; }
    
};
bool Decreasing_Values (Pixel const& p1, Pixel const& p2){ return p1.value >= p2.value; }
bool Increasing_Keys (std::pair< double, Point >const& p1, std::pair< double, Point >const& p2) { return p1.first < p2.first; }

struct Decreasing { bool operator() (int i0, int i1) { return (i0 >= i1 ); } };

struct Decreasing_Double
{
    bool operator() (double const& v1, double const& v2) const { return v1 >= v2; }
};

struct Compare_Points
{
    bool operator() (cv::Point const& a, cv::Point const& b) const
    {
        if ( a.x < b.x ) return true;
        if ( a.x == b.x and a.y < b.y ) return true;
        return false;
    }
};

Point Root (std::map< Point, Point, Compare_Points >const& edgel_parents, Point p)
{
    return p;
}

void Print (std::vector<int>const& v)
{
    for ( auto e : v ) std::cout<<e<<" "<< std::endl;
}

void Print (std::vector<Vec3b>const& v)
{
    for ( auto e : v ) std::cout<<e<<" "<< std::endl;
}

void Plot (Mat& img, std::vector<int>const& v)
{
    for ( int i = 0; i < v.size(); i++ )
        circle( img, Point( i, 255 - v[i] ), 1, black, -1 );
}

void Plot (Mat& img, std::vector<Vec3b>const& v)
{
    for ( int i = 0; i < v.size(); i++ )
        for ( int k = 0; k < 3; k++ )
            circle( img, Point( i, v[i][k] ), 1, BGR[k], -1 );
}

bool Extract_Row (Mat const& image, int row, std::vector<int>& line)
{
    for ( int j = 0; j < image.cols; j++ )
        line.push_back( image.at<uchar>( row, j ) );
    return true;
}

bool Extract_Row (Mat const& image, int row, std::vector<Vec3b>& line)
{
    for ( int j = 0; j < image.cols; j++ )
        line.push_back( image.at<Vec3b>( row, j ) );
    return true;
}

void Plot_Row (std::string name, Mat const& image, int row)
{
    Mat img( 256, (int)image.cols, CV_8UC3, white );
    std::vector<Vec3b> line_color;
    //Extract_Row( image, row, line_color );
    //Plot( img, line_color );
    Mat image_gray;
    cv::cvtColor( image, image_gray, CV_BGR2GRAY );
    std::vector<int> line_gray;
    Extract_Row( image_gray, row, line_gray );
    //Print( line_gray );
    Plot( img, line_gray );
    imwrite( name, img );
}

void Sqrt (Mat const& x, Mat const& y, Mat& sqrt)
{
    sqrt = Mat( x.rows, x.cols, CV_64F );
    for ( int i = 0; i < sqrt.rows; i++ )
        for ( int j = 0; j < sqrt.cols; j++ )
            sqrt.at<double>( i, j ) = std::sqrt( pow( x.at<double>(i,j), 2 ) + pow( y.at<double>(i,j), 2 ) ); // short = CV_16S
}

void Draw_Gradients (std::multimap< double, Point >const& edgels, Mat const& image_dx, Mat const& image_dy, int scale_factor, Mat& image_grad)
{
    resize( image_grad, image_grad, image_grad.size() * scale_factor );
    for ( auto it = edgels.begin(); it != edgels.end(); it++ )
    {
        Point p = it->second;
        int x = image_dx.at<double>( p ) * scale_factor * 10;
        int y = image_dy.at<double>( p ) * scale_factor * 10;
        line( image_grad, scale_factor * p, scale_factor * p + Point( x, y ), Green, 1 );
    }
    for ( auto it = edgels.begin(); it != edgels.end(); it++ )
        circle( image_grad, scale_factor * it->second, 1, Blue, -1 );
}

bool Acceptable_Edgel_Small_Corner (Point pixel, Mat_<bool>const& mask, Point& edgel1, Point& edgel2)
{
    std::vector<Point> arrows{ Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
    for ( int k = 0; k < 4; k++ )
        if ( mask.at<bool>( pixel + arrows[ k ] ) and mask.at<bool>( pixel + arrows[ (k+1)%4 ] ) )
        {
            Point a0 = arrows[ (k+2)%4 ], a1= arrows[ (k+3)%4 ];
            if ( mask.at<bool>( pixel + a0 ) or mask.at<bool>( pixel + a1 ) or mask.at<bool>( pixel + a0 + a1 ) ) return true; // acceptable edgel
            edgel1 = pixel + arrows[ k ];
            edgel2 = pixel + arrows[ (k+1)%4 ];
            return false;
        }
    return true;
}

bool Acceptable_Edgel_Small_Corner (Point pixel, Mat_<bool>const& mask)
{
    std::vector<Point> arrows{ Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };
    for ( int k = 0; k < 4; k++ )
        if ( mask.at<bool>( pixel + arrows[ k ] ) and mask.at<bool>( pixel + arrows[ (k+1)%4 ] ) )
        {
            Point a0 = arrows[ (k+2)%4 ], a1= arrows[ (k+3)%4 ];
            if ( mask.at<bool>( pixel + a0 ) or mask.at<bool>( pixel + a1 ) or mask.at<bool>( pixel + a0 + a1 ) ) return true; // acceptable edgel
            return false; // unacceptable edgel = small corner
        }
    return true;
}

bool Acceptable_Edgel_Big_Corner (Point pixel, Mat_<bool>const& mask)
{
    std::vector<Point> arrows_diag{ Point(1,1), Point(-1,1), Point(-1,-1), Point(1,-1) };
    for ( int k = 0; k < 4; k++ )
    {
        Point a0 = arrows_diag[ k ], a1= arrows_diag[ (k+1)%4 ], a01 = 0.5 * ( a0 + a1 );
        if ( mask.at<bool>( pixel + a0 ) and mask.at<bool>( pixel + a1 ) and mask.at<bool>( pixel + a01 ))
        {
            if ( mask.at<bool>( pixel - a0 ) or mask.at<bool>( pixel - a1 ) or mask.at<bool>( pixel - a01 ) ) return true; // acceptable edgel
            return false; // unacceptable edgel = big corner
        }
    }
    return true;
}

bool Acceptable_Edgel (Point pixel, Mat_<bool>const& mask, Point& edgel1, Point& edgel2)
{
    if ( pixel.x == 0 or pixel.y == 0 or pixel.x+1 == mask.cols or pixel.y+1 == mask.rows ) return true;
    if ( ! Acceptable_Edgel_Big_Corner( pixel, mask ) ) return false;
    if ( ! Acceptable_Edgel_Small_Corner( pixel, mask, edgel1, edgel2 ) ) return false;
    return true;
}

bool Acceptable_Edgel (Point pixel, Mat_<bool>const& mask)
{
    if ( pixel.x == 0 or pixel.y == 0 or pixel.x+1 == mask.cols or pixel.y+1 == mask.rows ) return true;
    if ( ! Acceptable_Edgel_Small_Corner( pixel, mask ) ) return false;
    if ( ! Acceptable_Edgel_Big_Corner( pixel, mask ) ) return false;
    return true;
}

bool Try_Remove_Edgel (Point edgel, Mat_<bool>& live_mask, int& removed_edgels)
{
    Point edgel1, edgel2;
    if ( Acceptable_Edgel( edgel, live_mask, edgel1, edgel2 ) ) return false;
    //std::cout<<" -"<<edgel;
    live_mask.at<bool>( edgel ) = false;
    removed_edgels++;
    Try_Remove_Edgel( edgel1, live_mask, removed_edgels );
    Try_Remove_Edgel( edgel2, live_mask, removed_edgels );
    return true;
}

bool Find_Edgels (Mat const& image_magnitude, double edgels_ratio, std::multimap< double, Point >& edgels)
{
    // Order all pixels according their magnitudes
    int num_pixels = image_magnitude.rows * image_magnitude.cols;
    Mat_<bool> edgels_mask( image_magnitude.rows, image_magnitude.cols, false );
    Mat_<bool> live_mask = edgels_mask.clone();
    std::vector< std::pair< double, Point > > pixels;
    for ( int i = 0; i < image_magnitude.rows; i++ )
        for ( int j = 0; j < image_magnitude.cols; j++ )
            pixels.push_back( std::make_pair( image_magnitude.at<double>( i, j ), Point( j, i ) ) );
    sort( pixels.begin(), pixels.end(), Increasing_Keys );
    for ( int k = num_pixels-1; k >= 0 and edgels.size() < edgels_ratio * num_pixels; k-- )
    {
        edgels_mask.at<bool>( pixels[k].second ) = true;
        if ( Acceptable_Edgel_Small_Corner( pixels[k].second, edgels_mask ) )
        {
            edgels.insert( pixels[k] );
            live_mask.at<bool>( pixels[k].second ) = true;
        }
        pixels.erase( pixels.begin() + k );
    }
    //std::cout<<" pixels="<<pixels.size();
    
    // removing superfluous edgels
    Point edgel1, edgel2; // empty variables needed for Acceptable_Edgel
    int removed_edgels = 0;
    //
    for ( auto it = edgels.begin(); it != edgels.end(); it++ ) // over initial edgels in the increasing order of magnitude
    {
        if ( live_mask.at<bool>( it->second ) )
            Try_Remove_Edgel( it->second, live_mask, removed_edgels ); // recursively, live_mask can become smaller
        while ( pixels.size() > 0 and removed_edgels > 0 )
        {
            Point p = pixels.rbegin()->second; // strongest pixel that isn't an edgel
            if ( !edgels_mask.at<bool>( p ) and Acceptable_Edgel( p, edgels_mask, edgel1, edgel2 ) )
            {
                //std::cout<<" +"<<p;
                live_mask.at<bool>( p ) = true;
                edgels_mask.at<bool>( p ) = true; // mark strongest pixels from the remaining map as edgels
                removed_edgels--;
            }
            pixels.erase( pixels.begin() + (int)pixels.size()-1 );
        }
    }//
    std::cout<<"Tried="<<num_pixels-(int)pixels.size()<< std::endl;
    edgels.clear();
    for ( int i = 0; i < image_magnitude.rows; i++ )
        for ( int j = 0; j < image_magnitude.cols; j++ )
            if ( live_mask.at<bool>( i, j ) )
                edgels.insert( std::make_pair( image_magnitude.at<double>( i, j ), Point( j, i ) ) );
    //
    return true;
}

bool Draw_Histogram (std::vector<double>const& histogram, Mat& image)
{
    return true;
}

bool Variance (std::vector<double>const& sums, double& var)
{
    if ( sums[0] == 0 ) return false;
    double m = (double)sums[1] / sums[0];
    var = double ( sums[2] - 2 * m * sums[1] ) / sums[0] + m * m;
    return true;
}

bool Otsu_Threshold (Mat const& image, int& threshold_value)
{
    bool print = false;
    // Histogram
    int bound = 130;
    std::vector<long> histogram( 256, 0 );
    for ( int row = 0; row < image.rows; row++ )
        for ( int col = 0; col < image.cols; col++ )
        {
            int v = (int)image.at<uchar>( row, col );
            if ( v < bound ) v = ( bound + v ) / 2;
            histogram[ v ]++;
        }
    //Otsu's binarization
    std::vector<double> sums0( 3, 0 ), sums1( 3, 0 );
    int length = (int)histogram.size();
    for ( int i = 0; i < length; i++ )
        for ( int p = 0; p < 3; p++ )
            sums1[ p ] += histogram[i] * pow( i, p );
    double cost, cost_min = big_constant, var0 = 0, var1 = 0;
    for ( int n = 0; n+1 < length; n++ ) // n+1 = number of values on the first class
    {
        if ( print ) std::cout<<"\nn="<<n<< std::endl;
        for ( int p = 0; p < 3; p++ )
        {
            int v = histogram[ n ] * pow( n, p );
            sums0[ p ] += v;
            sums1[ p ] -= v;
            if ( print ) std::cout<<" v="<<v<<" s0_"<<p<<"="<<sums0[p]<<" s1_"<<p<<"="<<sums1[p]<< std::endl;
        }
        if ( ! Variance( sums0, var0 ) ) continue;
        if ( ! Variance( sums1, var1 ) ) continue;
        cost = sums0[0] * var0 + sums1[0] * var1;
        if ( print ) std::cout<<" v0="<<var0<<" v1="<<var1<<" c="<<cost<< std::endl;
        if ( cost_min > cost ) { cost_min = cost; threshold_value = n; }
    }
    if ( print ) std::cout<<"\ncost_min="<<cost_min<<" threshold="<<threshold_value<< std::endl;
    //threshold( image, image_threshold, threshold_value, 255, THRESH_BINARY );
    return true;
}

bool Draw_Graphs (Mat const& image, std::multimap< Point, Graph, Compare_Points >const& pixels_graphs, Mat& image_graphs)
{
    for ( auto g : pixels_graphs )
    {
        int i = 0;
        for ( auto pair = vertices( g.second ); pair.first != pair.second; ++pair.first, i++)
        {
        //    circle( image, g.second[ *pair.first ], 1, colors[ i % colors.size() ], -1 );
            image_graphs.at<Vec3b>( g.second[ *pair.first ] ) = image.at<Vec3b>( g.second[ *pair.first ] );
        //circle( image, g.second[ *pair.first ], 1, black, -1 );
        }
    }
    return true;
}

bool Draw_Graph (Mat const& image, Graph const& graph, Mat& image_graphs)
{
    for ( auto pair = vertices( graph ); pair.first != pair.second; ++pair.first )
            image_graphs.at<Vec3b>( graph[ *pair.first ] ) = image.at<Vec3b>( graph[ *pair.first ] );
    return true;
}

bool Draw_Mask (Mat const& image, Mat const& mask, Mat& image_mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( row, col ) )
                image_mask.at<Vec3b>( row, col ) = image.at<Vec3b>( row, col );
    return true;
}

bool Save_Mask (Mat const& image, Mat const& mask, std::string name)
{
    Mat_<Vec3b> image_mask( image.size(), white );
    Draw_Mask( image, mask, image_mask );
    cv::imwrite( name, image_mask );
    return true;
}

bool Is_Boundary (Point p, Point size)
{
    if ( p.x == 0 or p.y == 0 or p.x == size.x-1 or p.y == size.y-1 ) return true;
    return false;
}

bool Is_Boundary (std::vector<Point>const& points, Point size)
{
    for ( auto p : points )
        if ( Is_Boundary( p, size ) ) return true;
    return false;
}

bool Neighbors (Mat_<bool>const& mask, Point point, std::vector<Point>const& shifts, int& num_neighbors, std::vector<bool>& neighbors)
{
    Point p;
    num_neighbors = 0;
    neighbors.assign( shifts.size(), true );
    for ( int i = 0; i < shifts.size(); i++ )
    {
        p = point + shifts[i];
        if ( p.x >= 0 and p.x < mask.cols and p.y >= 0 and p.y < mask.rows )
            neighbors[i] = mask.at<bool>( p ); // the presence of neighbor
        if ( neighbors[i] ) num_neighbors++;
    }
    return true;
}

/*
bool Remove_Isolated_Pixels (Mat_<bool>& mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( row, col ) )
                Remove_Isolated_Pixel( Point( col, row ), mask );
    return true;
}

bool Remove_Hanging_Pixel (Point point, Mat_<bool>& mask)
{
    if ( ! mask.at<bool>( point ) ) return false; // already removed
    Point p;
    int num_neighbors = 0;
    std::vector<bool> neighbors8( 8, true );
    Neighbors( mask, point, num_neighbors, neighbors8 );
    if ( num_neighbors != 1 ) return false;
    mask.at<bool>( point ) = false;
    for ( int i = 0; i < shifts8.size(); i++ )
    {
        if ( ! neighbors8[i] ) continue;
        p = point + shifts8[i];
        if ( p.x >= 0 and p.x < mask.cols and p.y >= 0 and p.y < mask.rows )
            if ( ! Remove_Isolated_Pixel( p, mask ) )
                Remove_Hanging_Pixel( p, mask );
    }
    return true;
}

bool Remove_Hanging_Pixels (Mat_<bool>& mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( row, col ) )
                Remove_Hanging_Pixel ( Point( col, row ), mask );
    return true;
}*/

bool Remove_Isolated_Pixel (Point point, Mat_<bool>& mask)
{
    if ( ! mask.at<bool>( point ) ) return false; // already removed
    int num_neighbors = 0;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    if ( num_neighbors == 0 ) { mask.at<bool>( point ) = false; return true; }
    return false;
}

bool Remove_External_Pixel (Point point, Mat_<bool>& mask)
{
    // exceptional cases
    if ( point.x < 0 or point.y < 0 or point.x >= mask.cols or point.y >= mask.rows ) return false;
    if ( Remove_Isolated_Pixel( point, mask ) ) return true;
    // corners aren't removed
    if ( point == Point(0,0) ) return false;
    if ( point == Point(0,mask.rows-1) ) return false;
    if ( point == Point(mask.cols-1,0) ) return false;
    if ( point == Point(mask.cols-1,mask.rows-1) ) return false;
    //bool debug = false; if ( point.x == mask.cols-1 ) debug = true;
    int num_neighbors;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    int changes = 0;
    for ( int i = 0; i < shifts8.size(); i++ )
        if ( neighbors[i] != neighbors[ (i+1) % neighbors.size() ] ) changes++;
    //if ( debug ) std::cout<<"\n"<<point<<" n="<<num_neighbors<<" c="<<changes;
    if ( changes == 2 and num_neighbors <= 4 )
    {
        mask.at<bool>( point ) = false;
        for ( int i = 0; i < shifts8.size(); i++ )
            if ( neighbors[i] )
                Remove_External_Pixel( point + shifts8[i], mask );
    }
    return true;
}

bool Remove_External_Pixels (Mat_<bool>& mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( Point( col, row ) ) )
                Remove_External_Pixel( Point( col, row ), mask );
    return true;
}

bool Add_Isolated_Pixel (Point point, Mat_<bool>& mask)
{
    if ( mask.at<bool>( point ) ) return false; // already added
    int num_neighbors = 0;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    if ( num_neighbors == neighbors.size() ) { mask.at<bool>( point ) = true; return true; }
    return false;
}

bool Add_Internal_Pixel (Point point, Mat_<bool>& mask)
{
    // exceptional cases
    if ( point.x < 0 or point.y < 0 or point.x >= mask.cols or point.y >= mask.rows ) return false;
    int num_neighbors;
    std::vector<bool> neighbors( shifts4.size(), true );
    Neighbors( mask, point, shifts4, num_neighbors, neighbors );
    if ( num_neighbors >= 3 ) // a pixel with at least 3 of 4 potential neighbors is called external
    {
        mask.at<bool>( point ) = true;
        neighbors.assign( shifts8.size(), true );
        Neighbors( mask, point, shifts8, num_neighbors, neighbors );
        for ( int i = 0; i < shifts8.size(); i++ )
            if ( neighbors[i] ) Remove_External_Pixel( point + shifts8[i], mask );
            else Add_Internal_Pixel( point + shifts8[i], mask );
    }
    return true;
}

bool Add_Internal_Pixels (Mat_<bool>& mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( ! mask.at<bool>( Point( col, row ) ) )
                Add_Internal_Pixel( Point( col, row ), mask );
    return true;
}

bool Pixels_to_Graph (Mat_<bool>const& mask, bool object, Graph& graph)
{
    graph.clear();
    std::map< Point, Graph::vertex_descriptor, Compare_Points > pixels_vertices;
    // Add vertices
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
        {
            if ( mask.at<bool>( row, col ) != object ) continue; // irrelevant pixel
            auto vertex = boost::add_vertex( graph );
            graph[ vertex ] = Point( col, row );
            pixels_vertices.insert( std::make_pair( Point( col, row ), vertex ) );
        }
    // Add horizontal edges
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col+1 < mask.cols; col++ )
        {
            Point p0( col, row ), p1(  col+1, row );
            if ( mask.at<bool>( p0 ) != object or mask.at<bool>( p1 ) != object ) continue;
            boost::add_edge( pixels_vertices[ p0 ], pixels_vertices[ p1 ], graph );
        }
    // Add vertical edges
    for ( int row = 0; row+1 < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
        {
            Point p0( col, row ), p1(  col, row+1 );
            if ( mask.at<bool>( p0 ) != object or mask.at<bool>( p1 ) != object ) continue;
            boost::add_edge( pixels_vertices[ p0 ], pixels_vertices[ p1 ], graph );
        }
    return true;
}

bool Boundary (Mat_<bool>const& mask, std::vector<bool>& boundary)
{
    int row = 0, col = 0;
    for ( row = 0, col = 0; row < mask.rows; row++ ) boundary.push_back( mask.at<bool>( row, col ) );
    for ( row = mask.rows-1, col = 0; col < mask.cols; col++ ) boundary.push_back( mask.at<bool>( row, col ) );
    for ( row = mask.rows-1, col = mask.cols-1; row >= 0; row-- ) boundary.push_back( mask.at<bool>( row, col ) );
    for ( row = 0, col = mask.cols-1; col >= 0; col-- ) boundary.push_back( mask.at<bool>( row, col ) );
    return true;
}
                      
bool Remove_Small_Components (bool object, int area_min, Mat_<bool>& mask, int& num_components)
{
    Graph graph;
    Pixels_to_Graph( mask, object, graph );
    std::vector<int> vertex_components( boost::num_vertices( graph ) );
    // Count the sizes of components
    num_components = boost::connected_components( graph, &vertex_components[0]);
    std::vector< int > component_sizes( num_components, 0 );
    for ( int i = 0; i != vertex_components.size(); ++i )
        component_sizes[ vertex_components[ i ] ]++;
    //std::cout<<"\nComponents:"; for ( int i = 0; i < component_sizes.size(); i++ ) std::cout<<" c"<<i<<"="<<component_sizes[i];
    // Select small components to remove
    std::map< int, std::vector<Point> > small_components;
    std::vector<Point> empty;
    if ( object ) area_min = *max_element( component_sizes.begin(), component_sizes.end() ); // keep only the largest foreground object
    for ( int i = 0; i < component_sizes.size(); i++ )
        if ( component_sizes[i] < area_min ) small_components.insert( std::make_pair( i, empty ) );
    // Mark pixels from small components
    for ( int i = 0; i != vertex_components.size(); ++i )
    {
        auto it = small_components.find( vertex_components[ i ] );
        if ( it != small_components.end() ) (it->second).push_back( graph[i] );
    }
    num_components -= (int)small_components.size();
    // Check if any small components touches the boundary
    if ( object ) for ( auto it = small_components.begin(); it != small_components.end(); it++ )
        if ( Is_Boundary( it->second, Point( mask.cols, mask.rows ) ) ) // keep components touching the boundary
        {
            (it->second).clear();
            num_components++;
        }//
    // Remove superfluos pixels
    for ( auto it = small_components.begin(); it != small_components.end(); it++ )
        for ( auto p : it->second ) mask.at<bool>( p ) = ! object;
    /*
    if ( object ) return true;
    sort( component_sizes.begin(), component_sizes.end() );
    std::multimap< int, int > differences;
    for ( int i = 1; i < component_sizes.size(); i++ )
    {
        std::cout<<" c"<<i<<"="<<component_sizes[i];
        differences.insert( std::make_pair( component_sizes[i] - component_sizes[i-1], i ) );
    }
    std::cout<<"\nDifferences:"; for ( auto  d : differences ) std::cout<<" d="<<d.first<<" i="<<d.second;
     */
    return true;
}

// Count boundary vertices
bool Boundary_Vertices (Mat_<bool>const& mask, int& boundary_vertices)
{
    boundary_vertices = 0;
    std::vector<bool> boundary;
    Boundary( mask, boundary );
    int changes = 0;
    for ( int i = 0; i < boundary.size(); i++ )
        if ( boundary[i] != boundary[ (i+1) % boundary.size() ] ) changes++;
    if ( changes % 2 != 0 )
    {
        std::cout<<"\nError in Remove_Small_Components"<< std::endl;
        return false;
    }
    boundary_vertices = int( changes / 2 );
    return true;
}
        
class Method
{
public:
    // parameters
    bool test = false;
    int size_small = 200;
    double area_min = 100;
    std::vector<int> min_areas;
    Point image_sizes;
    String input_folder, name_base, ext, output_folder, name;
    std::multimap< int, Point > values_pixels;
    //std::multimap< Point, Graph, Compare_Points > pixels_graphs;
    Mat_<bool> mask;
    cv::Mat image;
    
    Method (String _input_folder, String _name_base, String _ext, String _output_folder)
    {
        input_folder = _input_folder;
        name_base = _name_base;
        ext = _ext;
        output_folder = _output_folder;
        for ( int i = 1; i < 6; i++ ) min_areas.push_back( 100* i );
    }
    
    bool Threshold (Mat& image)
    {
        Point grid_sizes ( 1 + int( image.cols / size_small ), 1 + int( image.rows / size_small ) );
        Mat_<int> thresholds( grid_sizes.y, grid_sizes.x );
        Mat_<Point> shifts( grid_sizes.y, grid_sizes.x );
        for ( int row = 0; row < shifts.rows; row++ )
            for ( int col = 0; col < shifts.cols; col++ )
                shifts.at<Point>( row, col ) = Point( col, row ) * size_small;
        //int threshold_value = 0;
        values_pixels.clear();
        for ( int i = 0; i < grid_sizes.y; i++ )
            for ( int j = 0; j < grid_sizes.x; j++ )
            {
                int size_x = min( size_small, image.cols - shifts( i, j ).x );
                int size_y = min( size_small, image.rows - shifts( i, j ).y );
                //std::cout<<"\nr="<<row<<" c="<<col<<" s="<<shifts( row, col )<<" size_x="<<size_x<<" size_y="<<size_y;
                Mat image_small = image( Rect( shifts( i, j ).x, shifts( i, j ).y, size_x, size_y ) );
                Otsu_Threshold( image_small, thresholds( i, j ) );
                for ( int row = 0; row < image_small.rows; row++ )
                    for ( int col = 0; col < image_small.cols; col++ )
                    {
                        int v = (int)image_small.at<uchar>( row, col );
                        if ( v <= thresholds( i, j ) ) values_pixels.insert( std::make_pair( v, shifts( i, j ) + Point( col, row ) ) );
                    }
                //std::cout<<"\ni="<<i<<" j="<<j<<" t="<<threshold_value;
                //std::cout<<" "<<thresholds( i, j );
            }
        return true;
    }
    
    /*
    bool Pixels_to_Graphs ()
    {
        std::map< Point, Point, Compare_Points > pixels_parents;
        for ( auto v : values_pixels )
        {
            Point p = v.second;
            //std::cout<<" "<<v.first;
            auto p_it = pixels_parents.find( p );
            if ( p_it == pixels_parents.end() ) // new pixel generates a graph
            {
                pixels_parents.insert( std::make_pair( p, p ) );
                Graph graph;
                auto vertex = boost::add_vertex( graph );
                graph[ vertex ] = p;
                pixels_graphs.insert( std::make_pair( p, graph ) );
                //std::cout<<"*";
                continue;
            }
        }
        Mat_<Vec3b> image_graphs( image_sizes, white );
        Draw_Graphs( image, pixels_graphs, image_graphs );
        cv::imwrite( name + "_graphs" + std::to_string( pixels_graphs.size() ) + ".png", image_graphs );
        return true;
    }*/
    
    bool Image_to_Graph (int image_ind)
    {
        bool debug = true, save_images = true; //false;
        if ( debug ) save_images = true;
        name = name_base + std::to_string( image_ind );
        std::cout<<"\n"<<name<< std::endl;
        image = cv::imread( input_folder + name + "." + ext, CV_LOAD_IMAGE_COLOR );
        if ( !image.data ) { std::cout<<" not found"<< std::endl; return false; }
        if ( test ) image = image( cv::Rect( 0, 0, size_small, size_small ) );
        image_sizes = Point( image.cols, image.rows );
        name = output_folder + name;
        
        //int num_pixels = image.rows * image.cols;
        
        // Convert to grayscale
        cv::Mat image_gray;
        cv::cvtColor( image, image_gray, CV_BGR2GRAY );
        
        // Thresholding
        Threshold( image_gray );
        mask = cv::Mat_<bool>( image_sizes.y, image_sizes.x, false );
        for ( auto v : values_pixels ) mask.at<bool>( v.second ) = true;
        //if ( save_images ) Save_Mask( image, mask, "_split" + std::to_string( size_small ) + ".png" );
        if ( debug ) std::cout<<" r="<<(double)values_pixels.size() / ( image.rows * image.cols )<< std::endl;
        
        // Smooth boundary contours
        Remove_External_Pixels( mask );
        //if ( save_images ) Save_Mask( image, mask, name + "_RE.png" );
        Add_Internal_Pixels( mask );
        //if ( save_images ) Save_Mask( image, mask, name + "_AI.png" );
        
        // Remove small objects
        std::vector<int> num_vertices( min_areas.size() ), num_domains( min_areas.size() ), num_walls( min_areas.size() );
        for ( int i = 0; i < min_areas.size(); i++ )
        {
            Remove_Small_Components( true, min_areas[i], mask, num_walls[i] ); // true means foreground
            //if ( debug ) std::cout<<" walls="<<num_walls[i];
            //if ( save_images ) Save_Mask( image, mask, name + "_connected" + std::to_string( min_areas[i] ) + ".png" );
        
            // Remove small holes
            Remove_Small_Components( false, min_areas[i], mask, num_domains[i] ); // false means background
            Remove_External_Pixels( mask ); // could be sped up
            if ( save_images ) Save_Mask( image, mask, name + "_no_holes" + std::to_string( min_areas[i] ) + ".png" );
        
            // Subtract boundary vertices
            int boundary_vertices = 0;
            Boundary_Vertices( mask, boundary_vertices );
            if ( debug ) std::cout<<" b="<<boundary_vertices;
            num_vertices[i] = 2*num_domains[i] - boundary_vertices - 1;
            std::cout<<" ("<< min_areas[i] << "," << num_domains[i] << "," <<num_vertices[i] <<")"<< std::endl;
        }
        /*
        values_pixels.clear();
        for ( int row = 0; row < mask.rows; row++ )
            for ( int col = 0; col < mask.cols; col++ )
                if ( mask.at<bool>( row, col ) )
                    values_pixels.insert( std::make_pair( (int)image.at<uchar>( row, col ), Point( col, row ) ) );
        if ( debug ) std::cout<<" r="<<(double)values_pixels.size() / ( image.rows * image.cols );
        */
        return true;
    }
};

int main ()
{
    String input_folder = "./input/";
    String output_folder = "./output/";
    String ext = "tiff"; //"jpeg";
    String name_base = "FC 100x"; //"vortex_image1";

    Method method( input_folder, name_base, ext, output_folder );
    //std::cout<<"area_min="<<method.area_min;
    std::cout<<"image (min_area, domains, vertices) ..."<< std::endl;
    
    for ( int image_ind = 2; image_ind < 4; image_ind++ )
        if ( ! method.Image_to_Graph( image_ind ) ) continue;
    return 1;
}
