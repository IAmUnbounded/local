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
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "tbb/tbb.h"
#include <ctime>

using namespace std;
namespace cv{
namespace face{
void FacemarkKazemiImpl :: readSplit(ifstream& is, splitr &vec)
{
    is.read((char*)&vec, sizeof(splitr));
}
void FacemarkKazemiImpl :: readLeaf(ifstream& is, vector<Point2f> &leaf)
{
    unsigned long size;
    is.read((char*)&size, sizeof(size));
    leaf.resize(size);
    is.read((char*)&leaf[0], leaf.size() * sizeof(Point2f));
}
void FacemarkKazemiImpl :: readPixels(ifstream& is,int index)
{
    is.read((char*)&loaded_pixel_coordinates[index][0], loaded_pixel_coordinates[index].size() * sizeof(Point2f));
}

bool FacemarkKazemiImpl :: load(string filename){
    if(filename.empty()){
        String error_message = "No filename found.Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    ifstream f(filename,ios::binary);
    if(!f.is_open()){
        String error_message = "No file with given name found.Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    cout<<"Loading....."<<endl;
    size_t len;
    f.read((char*)&len, sizeof(size_t));
    char* temp = new char[len+1];
    f.read(temp, len);
    temp[len] = '\0';
    string s(temp);
    delete [] temp;
    if(s.compare("cascade_depth")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    cout<<s<<":";
    unsigned long cascade_size;
    f.read((char*)&cascade_size,sizeof(cascade_size));
    cout<<cascade_size<<endl;
    loaded_forests.resize(cascade_size);
    f.read((char*)&len, sizeof(size_t));
    temp = new char[len+1];
    f.read(temp, len);
    temp[len] = '\0';
    s = string(temp);
    delete [] temp;
    if(s.compare("pixel_coordinates")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    loaded_pixel_coordinates.resize(cascade_size);
    unsigned long num_pixels;
    f.read((char*)&num_pixels,sizeof(num_pixels));
    for(unsigned long i=0;i<cascade_size;i++){
        loaded_pixel_coordinates[i].resize(num_pixels);
        readPixels(f,i);
    }
    f.read((char*)&len, sizeof(size_t));
    temp = new char[len+1];
    f.read(temp, len);
    temp[len] = '\0';
    s = string(temp);
    delete [] temp;
    if(s.compare("mean_shape")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    unsigned long mean_shape_size;
    f.read((char*)&mean_shape_size,sizeof(mean_shape_size));
    meanshape.resize(mean_shape_size);
    f.read((char*)&meanshape[0], meanshape.size() * sizeof(Point2f));
    if(!setMeanExtreme())
        exit(0);
    f.read((char*)&len, sizeof(size_t));
    temp = new char[len+1];
    f.read(temp, len);
    temp[len] = '\0';
    s =string(temp);
    cout<<s<<":";
    delete [] temp;
    if(s.compare("num_trees")!=0){
        String error_message = "Data not saved properly.Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    unsigned long num_trees;
    f.read((char*)&num_trees,sizeof(num_trees));
    cout<<num_trees<<endl;
    for(unsigned long i=0;i<cascade_size;i++){
        for(unsigned long j=0;j<num_trees;j++){
            regtree tree;
            f.read((char*)&len, sizeof(size_t));
            char* temp2 = new char[len+1];
            f.read(temp2, len);
            temp2[len] = '\0';
            s =string(temp2);
            delete [] temp2;
            if(s.compare("num_nodes")!=0){
                String error_message = "Data not saved properly.Aborting.....";
                CV_Error(Error::StsBadArg, error_message);
                return false;
            }
            unsigned long num_nodes;
            f.read((char*)&num_nodes,sizeof(num_nodes));
            for(unsigned long k=0;k<num_nodes;k++){
                f.read((char*)&len, sizeof(size_t));
                char* temp3 = new char[len+1];
                f.read(temp3, len);
                temp3[len] = '\0';
                s =string(temp3);
                delete [] temp3;
                tree_node node;
                if(s.compare("split")==0){
                    splitr split;
                    readSplit(f,split);
                    node.split = split;
                    node.leaf.clear();
                }
                else if(s.compare("leaf")==0){
                    vector<Point2f> leaf;
                    readLeaf(f,leaf);
                    node.leaf = leaf;
                }
                else{
                    String error_message = "Data not saved properly.Aborting.....";
                    CV_Error(Error::StsBadArg, error_message);
                    return false;
                }
                tree.nodes.push_back(node);
            }
            loaded_forests[i].push_back(tree);
        }
    }
    f.close();
    return true;
}
/*bool FacemarkKazemiImpl::readSplit(FileNode& node,tree_node& node){
    FileNode n1 = node["split"];
    splitr split_;
    split_.index1 =(long) n1["index1"];
    split_.index2 =(long) n1["index2"];
    split_.thresh = (float)n1["thresh"];
    node.split = split_;
    node.leaf.clear();
}
bool FacemarkKazemiImpl::readLeaf(FileNode& node,tree_node& node){
    Mat leaf1 = (Mat)node["leaf"];
    vector<Point2f> leaf_(leaf1,leaf1+leaf1.cols);
    node.leaf = leaf_;
}
bool FacemarkKazemiImpl::readnodes(FileNode &node,regtree & tree){
    string s1 = string("node");
    unsigned long num_nodes = pow(2,tree_depth)-1;
    for(unsigned long k=0;k<num_nodes;k++){
        stringstream ss1;
        ss1<<k;
        s1=s1+ss1.c_str();
        FileNode n1 = n[s1];
        tree_node node;
        string type = (string)n1["type"];
        if(type.compare("split")==0)
            readSplit(n1,node);
        if(type.compare("leaf")==0)
            readLeaf(n1,node);
        tree.nodes.push_back(node);
    }
}
bool FacemarkKazemiImpl :: readPixels(FileStorage & fs){
    Mat p=fs["pixel_coordinates"];
    vector<Point2f> temp(leaf1,leaf1+leaf1.cols);
}
bool FacemarkKazemiImpl::load(string filename){
    if(filename.empty()){
        String error_message = "No filename found.Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    FileStorage fs(filename, cv::FileStorage::READ_BASE64);
    cout<<"Loading....."<<endl;
    setTrainingParameters();
    readPixels();
    string s = string("tree");
    loaded_forests.resize(cascade_depth);
    for(unsigned long i=0;i<cascade_depth;i++){
        for(unsigned long j=0;j<num_trees_per_cascade_level;j++){
            stringstream ss;
            ss<<j;
            s=s+ss.c_str();
            FileNode n = fs[s];
            regtree tree;
            readNodes(n,tree);
            loaded_forests[i].push_back(tree);
        }
    }
    return true;
}
*/

bool FacemarkKazemiImpl::getShape(Mat image,vector<Rect> faces, vector< vector<Point2f> >& shapes){
    if(image.empty()){
        String error_message = "No image found.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(faces.empty()){
        String error_message = "No faces found.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(meanshape.empty()||loaded_forests.empty()||loaded_pixel_coordinates.empty()){
        String error_message = "Model not loaded properly.Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(loaded_forests.size()==0){
        String error_message = "Model not loaded properly.Aboerting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(loaded_pixel_coordinates.size()==0){
        String error_message = "Model not loaded properly.Aboerting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    tree_node curr_node;
    vector<int> pixel_intensity;
    Mat warp_mat;
    for(unsigned long e=0;e<faces.size();e++){
        shapes[e]=meanshape;
        unnormalise(faces[e],warp_mat);
        for(unsigned long i=0;i<loaded_forests.size();i++){
            vector<Point2f> pixel_relative = loaded_pixel_coordinates[i];
            getRelativePixels(shapes[e],pixel_relative);
            getPixelIntensities(image,pixel_relative,pixel_intensity,faces[e]);
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
                    shapes[e][p]=shapes[e][p]+curr_node.leaf[p];
                }
            }
        }
        for(unsigned long i=0;i<shapes[e].size();i++){
            Point2f pt = shapes[e][i];
            Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
            Mat D = warp_mat*C;
            pt.x=float(D.at<double>(0,0));
            pt.y=float(D.at<double>(1,0));
            shapes[e][i]=pt;
        }
    }
    return true;
}
}//cv
}//face