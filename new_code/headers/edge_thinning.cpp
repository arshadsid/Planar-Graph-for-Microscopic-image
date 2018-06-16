/***********************************************************************************************************************
 * This file implements edge_thinning.cpp
***********************************************************************************************************************/

// C++ include files
#include <fstream>
#include <iostream>
#include <deque>
#include <set>
#include <string>
#include <iterator>
#include <utility>
#include <algorithm>
#include <limits>
#include <map>


// OpenCV include files
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;


// Boost include files
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

#include "edge_thinning.h"
#include "constants.h"

std::vector<Point> shifts8 = { Point(1,0), Point(1,1), Point(0,1), Point(-1,1), Point(-1,0), Point(-1,-1), Point(0,-1), Point(1,-1) };
std::vector<Point> shifts4 = { Point(1,0), Point(0,1), Point(-1,0), Point(0,-1) };


// Function to get the values of neighbours
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

bool Neighbors2 (Mat_<bool>const& mask, Point point, std::vector<Point>const& shifts, int& num_neighbors, std::vector<bool>& neighbors, Point *nbrs)
{
Point p;
num_neighbors = 0;
neighbors.assign( shifts.size(), true );
for ( int i = 0; i < shifts.size(); i++ )
{
p = point + shifts[i];
if ( p.x >= 0 and p.x < mask.cols and p.y >= 0 and p.y < mask.rows )
{
neighbors[i] = mask.at<bool>( p ); // the presence of neighbor
nbrs[i] = p;
}
if ( neighbors[i] ) num_neighbors++;
}
return true;
}


// Functions to remove an isolated pixel || where number of Neighbours is zero
bool Remove_Isolated_Pixel (Point point, Mat_<bool>& mask)
{
    if ( ! mask.at<bool>( point ) ) return false; // already removed
    int num_neighbors = 0;
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors( mask, point, shifts8, num_neighbors, neighbors );
    if ( num_neighbors == 0 ) { mask.at<bool>( point ) = false; return true; }
    return false;
}


// Remove an external pixel
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
    else{
        return false;
    }
    return true;
}

// Remove an external pixel
bool Remove_External_Pixel2 (Point point, Mat_<bool>& mask)
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
    Point nbrs[8];
    std::vector<bool> neighbors( shifts8.size(), true );
    Neighbors2( mask, point, shifts8, num_neighbors, neighbors, nbrs );
    int changes = 0;
    for ( int i = 0; i < shifts8.size(); i++ )
        if ( neighbors[i] != neighbors[ (i+1) % neighbors.size() ] ) changes++;
    //if ( debug ) std::cout<<"\n"<<point<<" n="<<num_neighbors<<" c="<<changes;
    if ( changes == 2 and num_neighbors == 5 )
    {

        bool connectivity = true;
        for(int cn = 0; cn < num_neighbors; cn++)
        {
            std::vector<bool> nbrs_neighbors( shifts8.size(), true );
            int num_nbrs_neighbors;
            Point nbrs_nbrs[8];
            Neighbors2( mask, nbrs[cn], shifts8, num_nbrs_neighbors, nbrs_neighbors, nbrs_nbrs );
            int connected = 0;
            for(int cnn = 0; cnn < num_nbrs_neighbors; cnn++)
            {
                for(int cnnn = 0; cnnn < num_neighbors; cnnn++)
                {
                    if (nbrs_nbrs[cnn] == nbrs[cnnn])
                        connected++;
                }
            }
            if(connected == 0 )connectivity = false;
        }
        if(connectivity == true)
        {
            mask.at<bool>( point ) = false;
            for ( int i = 0; i < shifts8.size(); i++ )
                if ( neighbors[i] )
                {
                    if(!Remove_External_Pixel( point + shifts8[i], mask ))
                        Remove_External_Pixel2( point + shifts8[i], mask );
                }
        }
    }
    return true;
}


// Function removes all external pixels
bool Remove_External_Pixels2 (Mat_<bool>& mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( Point( col, row ) ) )
                Remove_External_Pixel2( Point( col, row ), mask );
    return true;
}


// Function removes all external pixels
bool Remove_External_Pixels (Mat_<bool>& mask)
{
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( Point( col, row ) ) )
                Remove_External_Pixel( Point( col, row ), mask );
    return true;
}


// Function adds a missing internal pixel inside the edge
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

// Function adds all missing internal pixels inside the edges
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


// Check if pixel is a boundry pixel
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

// Remove all small components
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


Mat Save_Mask (Mat const& image, Mat const& mask, std::string name)
{
    Mat_<Vec3b> image_mask( image.size(), white );
    Draw_Mask( image, mask, image_mask );
    cv::imwrite( name, image_mask );
    return image_mask;
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


/****************************************|| EDGE_THINNING FUNCTION STARTS HERE ||***************************************
 *
 * @param image
 * @param mask
 * @param min_areas
 * @param save_images
 * @param name
 *
 * This function is called from main.cpp for edge thinning
 *
 * @return
 */
bool Edge_Thinning (Mat const& image, Mat_<bool>& mask, std::vector<int>& min_areas, bool& save_images, std::string name)
{
    // Smooth boundary contours
    Remove_External_Pixels( mask );
    if ( save_images ) Save_Mask( image, mask, name + "_RE.png" );
    Add_Internal_Pixels( mask );
    if ( save_images ) Save_Mask( image, mask, name + "_AI.png" );


    // Remove small objects
    std::vector<int> num_vertices( min_areas.size() ), num_domains( min_areas.size() ), num_walls( min_areas.size() );
    /*for ( int i = 0; i < min_areas.size(); i++ )
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
    }*/
    int i = 0;
    Remove_Small_Components( true, 100, mask, num_walls[i] ); // true means foreground
    //if ( debug ) std::cout<<" walls="<<num_walls[i];
    //if ( save_images ) Save_Mask( image, mask, name + "_connected" + std::to_string( min_areas[i] ) + ".png" );

    // Remove small holes
    Remove_Small_Components( false, 100, mask, num_domains[i] ); // false means background
    Remove_External_Pixels( mask ); // could be sped up
    Mat im;
    if ( save_images ) im = Save_Mask( image, mask, name + "_no_holes" + std::to_string( 100 ) + ".png" );

    // Subtract boundary vertices
    //int boundary_vertices = 0;
    //Boundary_Vertices( mask, boundary_vertices );
    //if ( debug ) std::cout<<" b="<<boundary_vertices;
    //num_vertices[i] = 2*num_domains[i] - boundary_vertices - 1;
    //std::cout<<" ("<< 100 << "," << num_domains[i] << "," <<num_vertices[i] <<")"<< std::endl;

    Remove_External_Pixels2( mask );
    if ( save_images ) Save_Mask( image, mask, name + "_output.png" );

    Mat_<Vec3b> image_mask( image.size(), white );
    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            image_mask.at<Vec3b>( row, col ) = image.at<Vec3b>( row, col );

    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( row, col ) )
                image_mask.at<Vec3b>( row, col ) = black;

    cv::imwrite(  name + "_output1.png" , image_mask );

    for ( int row = 0; row < mask.rows; row++ )
        for ( int col = 0; col < mask.cols; col++ )
            if ( mask.at<bool>( row, col ) )
                im.at<Vec3b>( row, col ) = black;

    cv::imwrite(  name + "_output2.png" , im );
    return true;
}