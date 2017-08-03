/*By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.*/
#ifndef __OPENCV_FACE_ALIGNMENTIMPL_HPP__
#define __OPENCV_FACE_ALIGNMENTIMPL_HPP__
#include "face_alignment.hpp"

namespace cv{
namespace face{
/**@brief structure determining split in regression tree
*/
struct splitr{
        //!index1 Index of the first coordinates among the test coordinates for deciding split.
        long index1;
        //! index2 index of the second coordinate among the test coordinates for deciding split.
        long index2;
        //! thresh threshold for deciding the split.
        float thresh;
};
/** @brief represents a node of the regression tree*/
struct node_info{
    //First pixel coordinate of split
    long index1;
    //Second pixel coordinate .split
    long index2;
    long depth;
    long node_no;
};
/** @brief regression tree structure. Each leaf node is a vector storing residual shape.
* The tree is represented as vector of leaves.
*/
struct tree_node{
    splitr split;
    std::vector<Point2f> leaf;
};
struct regtree{
    std::vector<tree_node> nodes;
};
/** @brief Represents a training sample
*It contains current shape, difference between actual shape
*and current shape. It also stores the image whose shape is being
*detected.
*/
struct training_sample{
    //! shapeResiduals vector which stores the residual shape remaining to be corrected.
    std::vector<Point2f> shapeResiduals;
    //! current_shape vector containing current estimate of the shape
    std::vector<Point2f> current_shape;
    //! actual_shape vector containing the actual shape of the face or the ground truth.
    std::vector<Point2f> actual_shape;
    //! image A mat object which stores the image.
    Mat image ;
    //! pixel_intensities vector containing pixel intensities of the coordinates chosen for testing
    std::vector<int> pixel_intensities;
    //! pixel_coordinates vector containing pixel coordinates used for testing
    std::vector<Point2f> pixel_coordinates;
    Rect bound;
};
class FacemarkKazemiImpl : public FacemarkKazemi{

public:

    virtual int getNumLandmarks() const ;
    virtual int getNumFaces() const ;
    FacemarkKazemiImpl(CascadeClassifier face_cascade);
    // Destructor for the class.
    virtual ~FacemarkKazemiImpl();
    bool getData(std::vector<String> filename,std::vector< std::vector<Point2f> > & trainlandmarks,std::vector<String> & trainimages);
    bool getMeanShapeRelative(std::vector<Rect> face,std::vector< std::vector<Point2f> > & initialshape);
    bool calcMeanShape(std::vector< std::vector<Point2f> > & trainlandmarks,std::vector<Mat>& trainimages,std::vector<Rect>& rectangles);
    bool scaleData(std::vector< std::vector<Point2f> > &  trainlandmarks,std::vector<Mat> & trainimages ,Size s=Size(200,200) );
    bool getFaces(Mat src,std::vector<Rect> &facep,double scaleFactor=1.05, int minNeighbors=3, int flags=0, Size minSize=Size(30,30), Size maxSize=Size());
    bool train(std::vector<Mat>& images, std::vector< std::vector<Point2f> >& landmarks,std::vector<Rect> rectangles,std::string filename,std::string modelFilename = std::string("Facemark_Kazemi.dat"));
    bool load(std::string filename);
    bool getShape(Mat image,std::vector<Rect> faces, std::vector< std::vector<Point2f> >& shapes);
protected:
    /** @brief This class is implementation of class FacemarkKazemi.
    *This class contains functions for training data and detection of
    *landmaks in images.
    */
    CascadeClassifier face_cascade_;
    /* numlandmarks stores number of landmarks to be detected in a  face.
      They are equal to the landmarks in images used for training.
      It is initialised to 194 as HELEN dataset contains 194 landmarks
      per image.
    */
    int numlandmarks;
    /* numfaces stores number of faces detected in an image
       Generally as in training data one face per image is detected*/
    int numfaces;
    float minmeanx;
    float maxmeanx;
    float minmeany;
    float maxmeany;
    /* meanshape This is a vector which stores the mean shape of all the images used in training*/
    std::vector<Point2f> meanshape;
    std::vector< std::vector<regtree> > loaded_forests;
    std::vector< std::vector<Point2f> > loaded_pixel_coordinates;
    // cascade_deapth This stores the deapth of cascade used for training.
    unsigned long cascade_depth;
    // tree_depth This stores the max height of the regression tree built.
    unsigned long tree_depth;
    // num_trees_per_cascade_level This stores number of trees fit per cascade level.
    unsigned long num_trees_per_cascade_level;
    // learning_rate stores the learning rate in gradient boosting, also reffered as shrinkage.
    float learning_rate;
    // oversampling_amount stores number of initialisations used to create training samples.
    unsigned long oversampling_amount;
    // num_test_coordinates stores number of test coordinates.
    unsigned long num_test_coordinates;
    // lambda stores a value to calculate probability of closeness of two coordinates.
    float lambda;
    // num_test_splits stores number of random test splits generated.
    unsigned long num_test_splits;
    /*Extract left node of the current node in the regression tree*/
    unsigned long left(unsigned long index);
    // Extract the right node of the current node in the regression tree
    unsigned long right(unsigned long index);
    // This function randomly  generates test splits to get the best split.
    splitr getTestSplits(std::vector<Point2f> pixel_coordinates,int seed);
    // This function writes a split node to the XML file storing the trained model
    /*bool writeSplit( FileStorage& fs, splitr split);
    // This function writes a leaf node to the XML file storing the trained model
    bool writeLeaf( FileStorage& fs,std::vector<Point2f> leaf);
    //This function writes training parameters to the XML file.
    bool writeParameters(FileStorage &fs);
    //This function writes nodes to a XML file.
    bool writeNodes(FileStorage& fs,std::vector<tree_node> nodes);
    // This function saves the trained model to the XML file
    bool saveModel(std::string filename,std::vector< std::vector<regtree> > forest,std::vector< std::vector<Point2f> > pixel_coordinates);
    */
    void writeSplit(std::ofstream& os,const splitr split);
    // This function writes a leaf node to the binary file storing the trained model
    void writeLeaf(std::ofstream& os, const std::vector<Point2f> &leaf);
    // This function writes a tree to the binary file containing the model
    void writeTree(std::ofstream &f,regtree tree);
    // This function saves the pixel coordinates to a binary file
    void writePixels(std::ofstream& f,std::vector<Point2f> pixel_coordinates); 
    // This function saves model to the binary file
    bool saveModel(std::string filename,std::vector< std::vector<regtree> > forest,std::vector< std::vector<Point2f> > pixel_coordinates);    //This function loads pixel coordinates
    void readPixels(std::ifstream& is,int index);
    //This function reads the split node of the tree from binary file
    void readSplit(std::ifstream& is, splitr &vec);
    //This function reads a leaf node of the tree.
    void readLeaf(std::ifstream& is, std::vector<Point2f> &leaf);
    /* This function generates pixel intensities of the randomly generated test coordinates used to decide the split.
    */
    bool getPixelIntensities(Mat img,std::vector<Point2f> pixel_coordinates_,std::vector<int>& pixel_intensities_,Rect face);
    //This function initialises the training parameters.
    bool setTrainingParameters(std::string filename);
    bool unnormalise(Rect r,Mat &warp);
    bool normalise(Rect r,Mat &warp);
    // This function gets the relative position of the test pixel coordinates relative to the current shape.
    // This function gets the landmarks in the meanshape nearest to the pixel coordinates.
    unsigned long getNearestLandmark (Point2f pixels );
    bool getRelativePixels(std::vector<Point2f> sample,std::vector<Point2f>& pixel_coordinates_);
    // This function partitions samples according to the split
    unsigned long divideSamples (splitr split,std::vector<training_sample>& samples,unsigned long start,unsigned long end);
    // This function fits a regression tree according to the shape residuals calculated to give weak learners for GBT algorithm.
    bool buildRegtree(regtree &tree,std::vector<training_sample>& samples,std::vector<Point2f> pixel_coordinates);
    // This function greedily decides the best split among the test splits generated.
    bool getBestSplit(std::vector<Point2f> pixel_coordinates, std::vector<training_sample>& samples,unsigned long start ,
                                        unsigned long end,splitr& split,std::vector< std::vector<Point2f> >& sum,long node_no);
    // This function randomly generates test coordinates for each level of cascade.
    void getTestCoordinates (std::vector< std::vector<Point2f> >& pixel_coordinates,float min_x,float min_y, float max_x , float max_y);
    // This function implements gradient boosting by fitting regression trees
    std::vector<regtree> gradientBoosting(std::vector<training_sample>& samples,std::vector<Point2f> pixel_coordinates);
    // This function creates training sample by randomly assigning a current shape from set of shapes available.
    void createLeafNode(regtree& tree,long node_no,std::vector<Point2f> assign);
    // This function creates a split node in the regression tree.
    void createSplitNode(regtree& tree, splitr split,long node_no);
    ///bool getRelativeShape(training_sample& sample);
    // This function prepares the training samples
    bool createTrainingSamples(std::vector<training_sample> &samples,std::vector<Mat> images,std::vector< std::vector<Point2f> > landmarks,
    std::vector<Rect> rectangle);   //This function generates a split
    bool generateSplit(std::queue<node_info>& curr,std::vector<Point2f> pixel_coordinates, std::vector<training_sample>& samples,
                                        splitr &split , std::vector< std::vector<Point2f> >& sum);
    bool setMeanExtreme();
    //friend class getRelShape;
    //friend class getRelPixels;
};
}//face
}//cv
#endif
