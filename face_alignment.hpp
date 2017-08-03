/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#ifndef __OPENCV_FACE_ALIGNMENT_HPP__
#define __OPENCV_FACE_ALIGNMENT_HPP__

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <queue>
#include <algorithm>
#include <fstream>

namespace cv{
namespace face{

class FacemarkKazemi
{
public:
        /** @returns The number of landmarks detected in an image */
        virtual int getNumLandmarks() const = 0;
        /** @returns The number of faces detected in an image */
        virtual int getNumFaces() const = 0;
        /** @brief This function extract the data for training from .txt files which contain the corresponding image name and landmarks.
        *Each file's first line should give the path of the image whose
        *landmarks are being described in the file.Then in the subsequent
        *lines there should be coordinates of the landmarks in the image
        *i.e each line should be of the form "x,y"(ignoring the double quotes)
        *where x represents the x coordinate of the landmark and y represents
        *the y coordinate of the landmark.
        *
        *For reference you can see the files as provided in the
        *<a href="http://www.ifp.illinois.edu/~vuongle2/helen/">HELEN dataset</a>
        * @param filename A vector containing name of the .txt files.
        * @param trainlandmarks A vector that would store shape or landmarks of all images.
        * @param trainimages A vector which stores the images whose landmarks are tracked
        * @returns A boolean value
        */
        virtual bool getData(std::vector<String> filename,std::vector<std::vector<Point2f> > & trainlandmarks
                ,std::vector<String> & trainimages)=0;
        /** @brief This function gets the relative shape of the face scaled and centered
        according to the bounding rectangle of the detected face.
        * @param face A vector storing rectangles of all  faces in the current image.
        * @param initialshape A vector storing initial shapes of all the faces in the current image.
        * @returns A boolean value
        */
        //virtual bool getMeanShapeRelative(std::vector<Rect> face,std::vector< std::vector<Point2f> > & initialshape)=0;
        /** @brief This function gets the bounding rectangle of all the faces in an image
        * @param src  stores the image in which faces have ti be detected
        * @param facep  stores the bounding boxes of the face whose initial shape has to be found out
        * choose 0 to select HAAR cascade classifier.
        * choose 1 to select LBP cascade classifier.
        * @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
        * @param minNeighbors  Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        * @param flags Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects.
        * It is not used for a new cascade.
        * @param minSize Minimum possible object size. Objects smaller than that are ignored.
        * @param maxSize Maximum possible object size. Objects larger than that are ignored.
        * Refer to face detection module for other parameters. the parameters are same as detectMultiScale
        * function in face detection module
        * @returns A boolean value.
        * @sa cv::CascadeClassifier::detectMultiScale
        */
        virtual bool getFaces(Mat src,std::vector<Rect> &facep,double scaleFactor=1.05
        , int minNeighbors=3, int flags=0, Size minSize=Size(30,30), Size maxSize=Size())=0;
        /** @brief This function calculates mean shape while training.
        * This function is only called when new training data is supplied by the train function.
        *@param trainlandmarks This stores the landmarks of corresponding images.
        *@param trainimages This stores the images which serve as training data.
        *@returns A boolean value
        */
        virtual bool calcMeanShape(std::vector< std::vector<Point2f> > & trainlandmarks,std::vector<Mat>& trainimages,std::vector<Rect>& faces)=0;
        /** @brief This functions scales the annotations to a common size which is considered same for all images.
        * @param trainlandmarks stores the landmarks of the corresponding training images.
        * @param trainimages stores the images which are to be scaled.
        * @param s stores the common size to which all the images are scaled.
        * @returns A boolean value
        */
        virtual bool scaleData(std::vector< std::vector<Point2f> > & trainlandmarks,
                                    std::vector<Mat> & trainimages , Size s=Size(460,460) )=0;
        /** @brief This function is used to train the model using gradient boosting to get a cascade of regressors
        *which can then be used to predict shape.
        *@param images stores the images which are used in training samples.
        *@param landmarks stores the landmarks detected in a particular image.
        *@param configfile stores parameters for training model
        *@returns A boolean value
        */
        virtual bool train(std::vector<Mat>& images, std::vector< std::vector<Point2f> >& landmarks,std::vector<Rect> rectangles,std::string configfile,std::string modelFilename= std::string("Facemark_Kazemi.dat"))=0;
        virtual bool load(std::string filename)=0;
        virtual bool getShape(Mat image,std::vector<Rect> faces, std::vector< std::vector<Point2f> >& shapes)=0;
        virtual ~FacemarkKazemi();
};
CV_EXPORTS Ptr<FacemarkKazemi> createFacemarkKazemi(CascadeClassifier face_cascade);
}
}
#endif