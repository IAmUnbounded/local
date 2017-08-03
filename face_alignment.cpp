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
using namespace std;

namespace cv{
namespace face{

FacemarkKazemi::~FacemarkKazemi(){}
FacemarkKazemiImpl:: ~FacemarkKazemiImpl(){}
int FacemarkKazemiImpl:: getNumLandmarks() const {return numlandmarks;}
int FacemarkKazemiImpl:: getNumFaces() const {return numfaces;}
unsigned long FacemarkKazemiImpl::left(unsigned long index){
    return 2*index+1;
}
unsigned long FacemarkKazemiImpl::right(unsigned long index){
    return 2*index+2;
}
FacemarkKazemiImpl :: FacemarkKazemiImpl(CascadeClassifier face_cascade){
    face_cascade_= face_cascade;
    //initialise other variables
    numfaces =1;
    numlandmarks =194;
    //These variables are used for training data
    //These are initialised as described in the research paper
    //referenced above
    cascade_depth = 10;
    tree_depth = 4;
    num_trees_per_cascade_level = 500;
    learning_rate = float(0.1);
    oversampling_amount = 20;
    num_test_coordinates = 400;
    lambda = float(0.1);
    num_test_splits = 20;
    minmeanx=8000.0;
    maxmeanx=0.0;
    minmeany=8000.0;
    maxmeany=0.0;
}
bool FacemarkKazemiImpl:: getFaces(Mat src,vector<Rect> &facep,double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize){
    Mat frame_gray;
    cvtColor( src, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    vector<Rect> faces1;
    face_cascade_.detectMultiScale( frame_gray, faces1, scaleFactor,minNeighbors,flags,minSize,maxSize);
    if(faces1.size()==0)
        return false;
    facep=faces1;
    return true;
}
bool FacemarkKazemiImpl :: unnormalise(Rect r,Mat &warp){
    Point2f srcTri[3],dstTri[3];
    srcTri[0]=Point2f(0,0);
    srcTri[1]=Point2f(1,0);
    srcTri[2]=Point2f(0,1);
    dstTri[0]=Point2f(r.x,r.y);
    dstTri[1]=Point2f(r.x+r.width,r.y);
    dstTri[2]=Point2f(r.x,r.y+r.height);
    warp=getAffineTransform(srcTri,dstTri);
    return true;
}
bool FacemarkKazemiImpl :: normalise(Rect r,Mat &warp){
    Point2f srcTri[3],dstTri[3];
    dstTri[0]=Point2f(0,0);
    dstTri[1]=Point2f(1,0);
    dstTri[2]=Point2f(0,1);
    srcTri[0]=Point2f(r.x,r.y);
    srcTri[1]=Point2f(r.x+r.width,r.y);
    srcTri[2]=Point2f(r.x,r.y+r.height);
    warp=getAffineTransform(srcTri,dstTri);
    return true;
}
bool FacemarkKazemiImpl :: getData(vector<String> filename,vector< vector<Point2f> >
                              & trainlandmarks,vector<String> & trainimages)
{
    string img;
    vector<Point2f> temp;
    string s;
    string tok;
    vector<string> coordinates;
    ifstream f1;
    for(unsigned long j=0;j<filename.size();j++){
        f1.open(filename[j].c_str(),ios::in);
        if(!f1.is_open()){
            cout<<filename[j]<<endl;
            CV_Error(Error::StsError, "File can't be opened for reading!");
            return false;
        }
        //get the path of the image whose landmarks have to be detected
        getline(f1,img);
        //push the image paths in the vector
        trainimages.push_back(img);
        img.clear();
        while(getline(f1,s)){
            Point2f pt;
            stringstream ss(s); // Turn the string into a stream.
            while(getline(ss, tok,',')) {
                coordinates.push_back(tok);
                tok.clear();
            }
            pt.x=(float)atof(coordinates[0].c_str());
            pt.y=(float)atof(coordinates[1].c_str());
            coordinates.clear();
            temp.push_back(pt);
        }
        trainlandmarks.push_back(temp);
        temp.clear();
        f1.close();
    }
    return true;
}
/*
bool FacemarkKazemiImpl :: getMeanShapeRelative(vector<Rect> face,vector< vector<Point2f> > & initialshape){
    if(meanshape.empty()) {
            // throw error if no data (or simply return -1?)
            String error_message = "The data is not loaded properly by train function. Aborting...";
            CV_Error(Error::StsBadArg, error_message);
            return false;
    }
    Point2f srcTri[3];
    //source points to find warp matrix
    srcTri[0] = Point2f( minmeanx , minmeany );
    srcTri[1] = Point2f( maxmeanx, minmeany );
    srcTri[2] = Point2f( minmeanx, maxmeany );
    Point2f dstTri[3];
    for(unsigned long k=0;k<face.size();k++){
        vector<Point2f> temp;
        //destination points to which the image has to be warped
        dstTri[0] = Point2f(float(face[k].x) , float(face[k].y) );
        dstTri[1] = Point2f(float(face[k].x + face[k].width), float(face[k].y) );
        dstTri[2] = Point2f( float(face[k].x), float(face[k].y+face[k].height));
        //loop to calculate bounding rectangle of the mean shape found
        Mat warp_mat( 2, 3, CV_32FC1 );
        //get affine transform to calculate warp matrix
        warp_mat = getAffineTransform( srcTri, dstTri );
        //loop to initialize initial shape
        for(unsigned long i=0;i<meanshape.size();i++){
            Point2f pt1=meanshape[i];
            Mat C = (Mat_<double>(3,1) << pt1.x, pt1.y, 1);
            Mat D =warp_mat*C;
            pt1.x=float(abs(D.at<double>(0,0)));
            pt1.y=float(abs(D.at<double>(1,0)));
            temp.push_back(pt1);
        }
        initialshape.push_back(temp);
    }
    return true;
}*/
class doSum : public ParallelLoopBody
{
    public:
        doSum(vector<training_sample>* samples_,vector<Point2f>* sum_) :
        samples(samples_),
        sum(sum_)
        {
        }
        virtual void operator()( const cv::Range& range) const
        {
            for (size_t j = range.start; j <(size_t) range.end; ++j){
                for(unsigned long k=0;k<(*samples)[j].shapeResiduals.size();k++){
                    (*sum)[k]=(*sum)[k]+(*samples)[j].shapeResiduals[k];
                }
            }
        }
    private:
        vector<training_sample>* samples;
        vector<Point2f>* sum;
};
bool FacemarkKazemiImpl :: calcMeanShape (vector< vector<Point2f> >& trainlandmarks,vector<Mat>& trainimages,std::vector<Rect>& faces){
    //clear the loaded meanshape
    if(trainimages.empty()||trainlandmarks.size()!=trainimages.size()) {
        // throw error if no data (or simply return -1?)
        String error_message = "Number of images is not equal to corresponding landmarks. Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    meanshape.clear();
    Mat warp_mat( 2, 3, CV_64FC1 );
    Mat C,D;
    Point2f srcTri[3];
    Point2f dstTri[3];
    vector<Mat> finalm;
    vector< vector<Point2f> >finall;
    Mat dst=trainimages[0].clone();
    float xmean[200]={0.0};
    //array to store mean of y coordinates
    float ymean[200]={0.0};
    unsigned long k=0;
    //loop to calculate mean
    Mat src;
    for(unsigned long i=0;i<trainimages.size();i++){
        src = trainimages[i].clone();
        //get bounding rectangle of image for reference
        //function from facemark class
        vector<Rect> facesp;
        if(!getFaces(src,facesp)){
            continue;
        }
        Rect face = facesp[0];
        normalise(face,warp_mat);
        //loop to bring points to a common reference and adding
        for(k=0;k<trainlandmarks[i].size();k++){
            Point2f pt=trainlandmarks[i][k];
            C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
            D =warp_mat*C;
            pt.x=float(D.at<double>(0,0));
            pt.y=float(D.at<double>(1,0));
            trainlandmarks[i][k]=pt;
            xmean[k]=xmean[k]+pt.x;
            ymean[k]=ymean[k]+pt.y;
        }
        finalm.push_back(trainimages[i]);
        finall.push_back(trainlandmarks[i]);
        faces.push_back(face);
    }
    //dividing by size to get mean and initialize meanshape
    for(unsigned long i=0;i<k;i++){
        xmean[i]=xmean[i]/finalm.size();
        ymean[i]=ymean[i]/finalm.size();
        if(xmean[i]>maxmeanx)
            maxmeanx = xmean[i];
        if(xmean[i]<minmeanx)
            minmeanx = xmean[i];
        if(ymean[i]>maxmeany)
            maxmeany = ymean[i];
        if(ymean[i]<minmeany)
            minmeany = ymean[i];
        meanshape.push_back(Point2f(xmean[i],ymean[i]));
    }
    trainimages=finalm;
    trainlandmarks=finall;
    finalm.clear();
    finall.clear();
    return true;
}
bool FacemarkKazemiImpl :: setMeanExtreme(){
    if(meanshape.empty()){
        String error_message = "Model not loaded properly.No mean shape found.Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    for(unsigned long i=0;i<meanshape.size();i++){
        if(meanshape[i].x>maxmeanx)
            maxmeanx = meanshape[i].x;
        if(meanshape[i].x<minmeanx)
            minmeanx = meanshape[i].x;
        if(meanshape[i].y>maxmeany)
            maxmeany = meanshape[i].y;
        if(meanshape[i].y<minmeany)
            minmeany = meanshape[i].y;
    }
    return true;
}
bool FacemarkKazemiImpl :: scaleData( vector< vector<Point2f> > & trainlandmarks,
                                vector<Mat> & trainimages ,Size s)
{
    if(trainimages.empty()||trainimages.size()!=trainlandmarks.size()){
        // throw error if no data (or simply return -1?)
        String error_message = "The data is not loaded properly by train function. Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    float scalex,scaley;
    //scale all images and their landmarks according  to input size
    for(unsigned long i=0;i< trainimages.size();i++){
        //calculating scale for x and y axis
        scalex=float(s.width)/float(trainimages[i].cols);
        scaley=float(s.height)/float(trainimages[i].rows);
        resize(trainimages[i],trainimages[i],s);
        std::vector<Point2f> trainlandmarks_ = trainlandmarks[i];
        for (vector<Point2f>::iterator it = trainlandmarks_.begin(); it != trainlandmarks_.end(); it++) {
            Point2f pt= (*it);
            pt.x=pt.x*scalex;
            pt.y=pt.y*scaley;
            (*it)=pt;
        }
        trainlandmarks[i]=trainlandmarks_;
    }
    return true;
}
Ptr<FacemarkKazemi> createFacemarkKazemi(CascadeClassifier face_cascade){return Ptr<FacemarkKazemi>(new FacemarkKazemiImpl(face_cascade));}
}//cv
}//face