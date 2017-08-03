/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
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
the use of this software, even if advised of the possibility of such damage.
*/

#include "face_alignmentImpl.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <ctime>

using namespace std;

namespace cv{
namespace face{
// Threading helper classes
class getDiffShape : public ParallelLoopBody
{
    public:
        getDiffShape(vector<training_sample>* samples_) :
        samples(samples_)
        {
        }
        virtual void operator()( const cv::Range& range) const
        {
            for (size_t j = range.start; j < range.end; ++j){
                (*samples)[j].shapeResiduals.resize((*samples)[j].current_shape.size());
                for(unsigned long k=0;k<(*samples)[j].current_shape.size();k++)
                    (*samples)[j].shapeResiduals[k]=(*samples)[j].actual_shape[k]-(*samples)[j].current_shape[k];
            }
        }
    private:
        vector<training_sample>* samples;
};
/*
class getRelShape : public ParallelLoopBody
{
    public:
        getRelShape(vector<training_sample>* samples_,FacemarkKazemiImpl& object_) :
        samples(samples_),
        object(object_)
        {
        }
        virtual void operator()( const cv::Range& range) const
        {
            for (size_t j = range.start; j < range.end; ++j){
                object.getRelativeShape((*samples)[j].actual_shape,(*samples)[j].current_shape);
            }
        }
    private:
        vector<training_sample>* samples;
        FacemarkKazemiImpl& object;
};

class getRelPixels : public ParallelLoopBody
{
    public:
        getRelPixels(vector<training_sample>* samples_,FacemarkKazemiImpl& object_) :
        samples(samples_),
        object(object_)
        {
        }
        virtual void operator()( const cv::Range& range) const
        {
            for (size_t j = range.start; j < range.end; ++j){
                object.getRelativePixels(((*samples)[j]).current_shape,((*samples)[j]).pixel_coordinates);
            }
        }
    private:
        vector<training_sample>* samples;
        FacemarkKazemiImpl& object;
};*/

//This function initialises the training parameters.
bool FacemarkKazemiImpl::setTrainingParameters(string filename){
    cout << "Reading Training Parameters " << endl;
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    if (!fs.isOpened())
    {   String error_message = "Error while opening configuration file.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    int cascade_depth_;
    int tree_depth_;
    int num_trees_per_cascade_level_;
    float learning_rate_;
    int oversampling_amount_;
    int num_test_coordinates_;
    float lambda_;
    int num_test_splits_;
    fs["cascade_depth"]>> cascade_depth_;
    fs["tree_depth"]>> tree_depth_;
    fs["num_trees_per_cascade_level"] >> num_trees_per_cascade_level_;
    fs["learning_rate"] >> learning_rate_;
    fs["oversampling_amount"] >> oversampling_amount_;
    fs["num_test_coordinates"] >> num_test_coordinates_;
    fs["lambda"] >> lambda_;
    fs["num_test_splits"] >> num_test_splits_;
    cascade_depth = (unsigned long)cascade_depth_;
    tree_depth = (unsigned long) tree_depth_;
    num_trees_per_cascade_level = (unsigned long) num_trees_per_cascade_level_;
    learning_rate = (float) learning_rate_;
    oversampling_amount = (unsigned long) oversampling_amount_;
    num_test_coordinates = (unsigned  long) num_test_coordinates_;
    lambda = (float) lambda_;
    num_test_splits = (unsigned long) num_test_splits_;
    fs.release();
    cout<<"Parameters loaded"<<endl;
    return true;
}
void FacemarkKazemiImpl::getTestCoordinates (vector< vector<Point2f> >& pixel_coordinates,float min_x,
                                        float min_y, float max_x , float max_y)
{
    for (unsigned long i = 0; i < cascade_depth; ++i){
        vector<Point2f> temp;
        RNG rng(time(0));
        for (unsigned long j = 0; j < num_test_coordinates; ++j)
        {
            Point2f pt;
            pt.x = (float)rng.uniform(min_x,max_x);
            pt.y = (float)rng.uniform(min_y,max_y);
            temp.push_back(pt);
        }
        pixel_coordinates.push_back(temp);
    }
}

bool FacemarkKazemiImpl :: getRelativePixels(vector<Point2f> sample,vector<Point2f>& pixel_coordinates_){
    if(sample.size()!=meanshape.size()){
        String error_message = "Error while finding relative shape. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    Mat warp_mat;
    //get affine transform to calculate warp matrix
    warp_mat = estimateRigidTransform(sample,meanshape,false);
    for (vector<Point2f>::iterator it = pixel_coordinates_.begin(); it != pixel_coordinates_.end(); it++) {
        unsigned long index = getNearestLandmark(*it);
        //indexes.push_back(index);
        Point2f pt = (*it);
        pt = pt-meanshape[index];
        Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 0);
        if(!warp_mat.empty()){
            Mat D =warp_mat*C;
            pt.x=float((D.at<double>(0,0)));
            pt.y=float((D.at<double>(1,0)));
        }
        pt = pt+sample[index];
        (*it)=pt;
    }
    return true;
}
bool FacemarkKazemiImpl::getPixelIntensities(Mat img,vector<Point2f> pixel_coordinates_,vector<int>& pixel_intensities_,Rect face){
    if(pixel_coordinates_.size()==0){
        String error_message = "No pixel coordinates found. Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    Mat transform_mat;
    unnormalise(face,transform_mat);
    vector<Point2f> srcp,dstp;
    srcp.push_back(Point2f(0,0));
    srcp.push_back(Point2f(0,0));

    Mat src = img.clone();
    Mat dst = img.clone();
    Mat bgr[3];
    split(dst,bgr);
    for(int j=0;j<pixel_coordinates_.size();j++){
        Point2f pt = pixel_coordinates_[j];
        Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
        Mat D = transform_mat*C;
        pt.x=float(D.at<double>(0,0));
        pt.y=float(D.at<double>(1,0));
        pixel_coordinates_[j]=pt;
        //cv::circle(src,pt,5,cv::Scalar(0,0,255),CV_FILLED);
    }
    /*namedWindow("pappu");
    imshow("pappu",src);
    waitKey(0);*/
    for(unsigned long j=0;j<pixel_coordinates_.size();j++){
        int val;
        if(pixel_coordinates_[j].x>0&&pixel_coordinates_[j].x<img.cols&&pixel_coordinates_[j].y>0&&pixel_coordinates_[j].y<img.rows)
            val = ((int)bgr[0].at<uchar>(pixel_coordinates_[j])+(int)bgr[1].at<uchar>(pixel_coordinates_[j])+(int)bgr[2].at<uchar>(pixel_coordinates_[j]))/3;
        else
            val = 0;
        pixel_intensities_.push_back(val);
    }
    return true;
}
unsigned long FacemarkKazemiImpl::  getNearestLandmark(Point2f pixel)
{
    if(meanshape.empty()) {
            // throw error if no data (or simply return -1?)
            String error_message = "The data is not loaded properly by train function. Aborting...";
            CV_Error(Error::StsBadArg, error_message);
            return false;
    }
    float dist=float(1000000009.00);
    unsigned long index =0;
    for(unsigned long i=0;i<meanshape.size();i++){
        Point2f pt = meanshape[i]-pixel;
        if(sqrt(pt.x*pt.x+pt.y*pt.y)<dist){
            dist=sqrt(pt.x*pt.x+pt.y*pt.y);
            index = i;
        }
    }
    return index;
}
vector<regtree> FacemarkKazemiImpl::gradientBoosting(vector<training_sample>& samples,vector<Point2f> pixel_coordinates){
    vector<regtree> forest;
    for(unsigned long i=0;i<num_trees_per_cascade_level;i++){
            cout<<"Fit "<<i<<" trees"<<endl;
            regtree tree;
            buildRegtree(tree,samples,pixel_coordinates);
            forest.push_back(tree);
    }
    return forest;
}/*
bool FacemarkKazemiImpl:: getRelativeShape(vector<Point2f> actual_shape,vector<Point2f>& current_shape,float x,float y){
    if(actual_shape.size()!=current_shape.size()){
        String error_message = "Error while finding relative shape. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    float minx=8000.0,maxx=0.0,miny=8000.0,maxy=0.0;
    Point2f srcTri[3];
    for (vector<Point2f>::iterator it = current_shape.begin(); it != current_shape.end(); it++)
    {
        Point2f pt1;
        pt1.x=(*it).x;
        if(pt1.x<minx)
            minx=pt1.x;
        if(pt1.x>maxx)
            maxx=pt1.x;
        pt1.y=(*it).y;
        if(pt1.y<miny)
            miny=pt1.y;
        if(pt1.y>maxy)
            maxy=pt1.y;
    }
    //source points to find warp matrix
    srcTri[0] = Point2f(minx , miny);
    srcTri[1] = Point2f( maxx , miny);
    srcTri[2] = Point2f( minx , maxy);
    minx=8000.0;maxx=0.0;miny=8000.0;maxy=0.0;
    for (vector<Point2f>::iterator it = actual_shape.begin(); it != actual_shape.end(); it++)
    {
        Point2f pt1;
        pt1.x=(*it).x;
        if(pt1.x<minx)
            minx=pt1.x;
        if(pt1.x>maxx)
            maxx=pt1.x;
        pt1.y=(*it).y;
        if(pt1.y<miny)
            miny=pt1.y;
        if(pt1.y>maxy)
            maxy=pt1.y;
    }
    Point2f dstTri[3];
    RNG rng;
    dstTri[0] = Point2f(minx + x*(maxx-minx) , miny );
    dstTri[1] = Point2f( maxx - x*(maxx-minx), miny );
    dstTri[2] = Point2f( minx + x*(maxx-minx), maxy + y*(maxy-miny));
    Mat warp_mat( 2, 3, CV_32FC1 );
    //get affine transform to calculate warp matrix
    warp_mat = getAffineTransform( srcTri, dstTri );
    //loop to initialize initial shape
    for (vector<Point2f>::iterator it = current_shape.begin(); it !=current_shape.end(); it++) {
        Point2f pt = (*it);
        Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
        Mat D =warp_mat*C;
        pt.x=float((D.at<double>(0,0)));
        pt.y=float((D.at<double>(1,0)));
        (*it)=pt;
    }
    return true;
}*/
void FacemarkKazemiImpl :: writePixels(ofstream& f,vector<Point2f> pixel_coordinates){
    f.write((char*)&pixel_coordinates[0], pixel_coordinates.size() * sizeof(Point2f));
}
bool FacemarkKazemiImpl::createTrainingSamples(vector<training_sample> &samples,vector<Mat> images,vector< vector<Point2f> > landmarks,vector<Rect> rectangle){
    RNG rng;
    unsigned long in=0;
    samples.resize(oversampling_amount*images.size());
    for(unsigned long i=0;i<images.size();i++){
        for(unsigned long j=0;j<oversampling_amount;j++){
            RNG rng(in);
        //make the splits generated from randomly_generate_split function
            unsigned long  rindex=i;
            while(rindex==i)
                rindex =(unsigned long)rng.uniform(0,(int)landmarks.size()-1);
            samples[in].image=images[i];
            samples[in].actual_shape = landmarks[i];
            if(in%2==0)
                samples[in].current_shape = meanshape;
            else
                samples[in].current_shape = landmarks[rindex];
            samples[in].bound = rectangle[i];
            in++;
        }
    }
    //parallel_for_(Range(0,samples.size()),getRelShape(&samples,*this));
    parallel_for_(Range(0,samples.size()),getDiffShape(&samples));
    return true;
}
void FacemarkKazemiImpl :: writeLeaf(ofstream& os, const vector<Point2f> &leaf)
{
    unsigned long size = leaf.size();
    os.write((char*)&size, sizeof(size));
    os.write((char*)&leaf[0], leaf.size() * sizeof(Point2f));
}
void FacemarkKazemiImpl :: writeSplit(ofstream& os, splitr split)
{
    os.write((char*)&split, sizeof(split));
}
void FacemarkKazemiImpl :: writeTree(ofstream &f,regtree tree)
{
    string s("num_nodes");
    size_t len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long num_nodes = tree.nodes.size();
    f.write((char*)&num_nodes,sizeof(num_nodes));
    for(unsigned long i=0;i<tree.nodes.size();i++){
        if(tree.nodes[i].leaf.empty()){
            string s("split");
            size_t len = s.size();
            f.write((char*)&len, sizeof(size_t));
            f.write(s.c_str(), len);
            writeSplit(f,tree.nodes[i].split);
        }
        else{
            string s("leaf");
            size_t len = s.size();
            f.write((char*)&len, sizeof(size_t));
            f.write(s.c_str(), len);
            writeLeaf(f,tree.nodes[i].leaf);
        }
    }
}
/*
void FacemarkKazemiImpl :: writePixels(ofstream& f,vector<Point2f> pixel_coordinates){
    f.write((char*)&pixel_coordinates[0], pixel_coordinates.size() * sizeof(Point2f));
}
bool FacemarkKazemiImpl :: writeSplit( FileStorage& fs, splitr split)
{
    fs<<"split"<<"{";
    int index_1 = split.index1;
    int index_2 = split.index2;
    float thresh_ = split.thresh;
    fs<<"index1"<<index_1;
    fs<<"index2"<<index_2;
    fs<<"thresh"<<thresh_;
    fs<<"}";
    return true;
}
bool FacemarkKazemiImpl :: writeLeaf( FileStorage& fs,vector<Point2f> leaf){
    Mat leaf1(leaf);
    fs<<"leaf"<<leaf1;
    return true;
}
bool FacemarkKazemiImpl::writeParameters(FileStorage &fs){
    int cascade_depth_ = cascade_depth;
    int tree_depth_ = tree_depth;
    int num_trees_per_cascade_level_ = num_trees_per_cascade_level;
    float learning_rate_ = learning_rate;
    int oversampling_amount_ = oversampling_amount;
    int num_test_coordinates_ = num_test_coordinates;
    float lambda_ = lambda;
    int num_test_splits_ = num_test_splits;
    fs << "cascade_depth" << cascade_depth_;
    fs << "tree_depth"<< tree_depth_;
    fs << "num_trees_per_cascade_level" << num_trees_per_cascade_level_; 
    fs << "learning_rate" << learning_rate_;
    fs << "oversampling_amount" << oversampling_amount_;
    fs << "num_test_coordinates" << num_test_coordinates_;
    fs << "lambda" << lambda_;
    fs << "num_test_splits"<< num_test_splits_;
    return true;
}
bool FacemarkKazemiImpl::writeNodes(FileStorage& fs,vector<tree_node> nodes){
    string s1 = string("node");
    string type;
    for(unsigned long k=0;k<nodes.size();k++){
        stringstream ss1;
        ss1<<k;
        s1=s1+ss1.str();
        fs<<s1<<"{";
        tree_node node;
        if(nodes[k].leaf.empty()){
            type = string("split");
            fs<<"type"<<type;
            writeSplit(fs,nodes[k].split);
        }
        else{
            type = string("leaf");
            fs<<"type"<<type;
            writeLeaf(fs,nodes[k].leaf);
        }
        fs<<"}";
    }
    return true;
}
bool FacemarkKazemiImpl::saveModel(string filename,vector< vector<regtree> > forest,vector< vector<Point2f> > pixel_coordinates){
    FileStorage fs(filename, cv::FileStorage::WRITE_BASE64);
    writeParameters(fs);
    //cout<<"Done"<<endl;
    string s("parameter_list");
    fs << s;
    fs << "{";
    for (int i = 0; i < pixel_coordinates.size(); i++)
    {
        fs << s + "_" + to_string(i);
        vector<Point2f> tmp = pixel_coordinates[i];
        fs << tmp;
    }
    fs << "}";

    //cout<<"Done1"<<endl;
    s = string("tree");
    for(unsigned long i=0;i<forest.size();i++){
        for(unsigned long j=0;j<forest[i].size();j++){
            stringstream ss;
            ss<<j;
            s=s+ss.str();
            fs<<s<<"{";
            writeNodes(fs,forest[i][j].nodes);
            fs<<"}";
        }
    }
    fs.release();
    return true;
}*/
bool FacemarkKazemiImpl :: saveModel(string filename,vector< vector<regtree> > forest,vector< vector<Point2f> > pixel_coordinates){
    ofstream f(filename,ios::binary);
    if(!f.is_open()){
        String error_message = "Error while opening file to write model. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(forest.size()!=pixel_coordinates.size()){
        String error_message = "Incorrect training data. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    string s("cascade_depth");
    size_t len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long cascade_size = forest.size();
    f.write((char*)&cascade_size,sizeof(cascade_size));
    s =string("pixel_coordinates");
    len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long num_pixels = pixel_coordinates[0].size();
    f.write((char*)&num_pixels,sizeof(num_pixels));
    for(unsigned long i=0;i<pixel_coordinates.size();i++){
        writePixels(f,pixel_coordinates[i]);
    }
    s= string("mean_shape");
    len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long mean_shape_size = meanshape.size();
    f.write((char*)&mean_shape_size,sizeof(mean_shape_size));
    f.write((char*)&meanshape[0], meanshape.size() * sizeof(Point2f));
    s = string("num_trees");
    len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long num_trees = forest[0].size();
    f.write((char*)&num_trees,sizeof(num_trees));
    for(unsigned long i=0;i<forest.size();i++){
        for(unsigned long j=0;j<forest[i].size();j++){
            writeTree(f,forest[i][j]);
       }
    }
    return true;
}
bool FacemarkKazemiImpl::train(vector<Mat>& images, vector< vector<Point2f> >& landmarks,vector<Rect> rectangles,string filename,
                                string modelFilename){
    if(images.size()!=landmarks.size()){
        // throw error if no data (or simply return -1?)
        String error_message = "The data is not loaded properly. Aborting training function....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(!setTrainingParameters(filename)){
        String error_message = "Error while loading training parameters";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    vector<training_sample> samples;
    vector< vector<Point2f> > pixel_coordinates;
    getTestCoordinates(pixel_coordinates,minmeanx,minmeany,maxmeanx,maxmeany);
    createTrainingSamples(samples,images,landmarks,rectangles);
    images.clear();
    landmarks.clear();
    vector< vector<regtree> > forests;
    cout<<"Total Samples :"<<samples.size()<<endl;
    for(unsigned long i=0;i<cascade_depth;i++){
        for (std::vector<training_sample>::iterator it = samples.begin(); it != samples.end(); it++) {
            (*it).pixel_coordinates=pixel_coordinates[0];
        }
        //parallel_for_(Range(0,samples.size()),getRelPixels(&samples,*this));
        for (std::vector<training_sample>::iterator it = samples.begin(); it != samples.end(); it++){
            (*it).pixel_coordinates=pixel_coordinates[i];
            Mat src = (*it).image.clone();
            Mat transform_mat;
            unnormalise((*it).bound,transform_mat);
            cv::rectangle(src,(*it).bound,Scalar( 255, 0, 0 ));
            /*for(unsigned long k=0;k<(*it).current_shape.size();k++){
                Point2f pt = (*it).current_shape[k];
                Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
                Mat D = transform_mat*C;
                pt.x=float(D.at<double>(0,0));
                pt.y=float(D.at<double>(1,0));
                //cv::circle(src,pt,5,cv::Scalar(0,0,255),CV_FILLED);
            }
            namedWindow("pappu");
            imshow("pappu",src);
            waitKey(0);*/
            getRelativePixels((*it).current_shape,(*it).pixel_coordinates);
            getPixelIntensities((*it).image,(*it).pixel_coordinates,(*it).pixel_intensities,(*it).bound);
        }
        cout<<"got pixel intensities"<<endl;
        cout<<"Training "<<i<<" regressor"<<endl;
        vector<regtree> forest = gradientBoosting(samples,pixel_coordinates[i]);
        forests.push_back(forest);
    }
    saveModel(modelFilename,forests,pixel_coordinates);
    cout<<"Model saved"<<endl;
    //load(modelFilename);
    /*samples[0].current_shape = meanshape;
    getRelativeShape(samples[0].actual_shape,samples[0].current_shape);
    Mat src;
    src = samples[0].image.clone();
    /*for(unsigned long i=0;i<loaded_forests.size();i++){
        vector<Point2f> pixel_relative = loaded_pixel_coordinates[i];
        getRelativePixels(samples[0].current_shape,pixel_relative);
        vector<int> pixel_intensity;
        getPixelIntensities(samples[0].image,pixel_relative,pixel_intensity);
        for(unsigned long j=0;j<loaded_forests[i].size();j++){
            regtree tree = loaded_forests[i][j];
            tree_node curr_node = tree.nodes[0];
            unsigned long curr_node_index = 0;
            while(curr_node.leaf.size()==0)
            {
                if ((float)pixel_intensity[curr_node.split.index1] - (float)pixel_intensity[curr_node.split.index2] > curr_node.split.thresh)
                {
                    curr_node_index=left(curr_node_index);
                }
                else
                    curr_node_index=right(curr_node_index);
                curr_node = tree.nodes[curr_node_index];
            }
            for(unsigned long p=0;p<curr_node.leaf.size();p++){
                samples[0].current_shape[p]=samples[0].current_shape[p]+curr_node.leaf[p];
            }
        }
    }*/
    /*namedWindow("pappu2");
    imshow("pappu2",src);
    waitKey(0);
    samples[0].current_shape = meanshape;
    getRelativeShape(samples[0].actual_shape,samples[0].current_shape);
    //src = samples[0].image.clone();
    for(unsigned long i=0;i<forests.size();i++){
        vector<Point2f> pixel_relative = loaded_pixel_coordinates[i];
        getRelativePixels(samples[0].current_shape,pixel_relative);
        vector<int> pixel_intensity;
        getPixelIntensities(samples[0].image,pixel_relative,pixel_intensity);
        for(unsigned long j=0;j<forests[i].size();j++){
            regtree tree = forests[i][j];
            tree_node curr_node = tree.nodes[0];
            unsigned long curr_node_index = 0;
            while(curr_node.leaf.size()==0)
            {
                if ((float)pixel_intensity[curr_node.split.index1] - (float)pixel_intensity[curr_node.split.index2] > curr_node.split.thresh)
                {
                    curr_node_index=left(curr_node_index);
                }
                else
                    curr_node_index=right(curr_node_index);
                curr_node = tree.nodes[curr_node_index];
            }
            for(unsigned long p=0;p<curr_node.leaf.size();p++){
                samples[0].current_shape[p]=samples[0].current_shape[p]+curr_node.leaf[p];
            }
        }
    }*/
   /* src = samples[0].image.clone();
    for(unsigned long k=0;k<samples[0].current_shape.size();k++){
        cv::circle(src,samples[0].current_shape[k],5,cv::Scalar(0,0,255),CV_FILLED);
    }
    namedWindow("pappu1");
    imshow("pappu1",src);
    waitKey(0);
    cout<<"Done"<<endl;*/
    return true;
}
}//cv
}//face