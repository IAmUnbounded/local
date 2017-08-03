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
#include "face_alignment.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;

int main(int argc,char** argv){
    //Give the path to the directory containing all the files containing data
   CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | give the following arguments in following format }"
        "{ annotations a  |.     | (required) path to annotations txt file [example - /data/annotations.txt] }"
        "{ config c       |      | (required) path to configuration xml file containing parameters for training.[example - /data/config.xml] }"
        "{ model m        |      | (required) path to file containing trained model for face landmark detection[example - /data/model.dat] }"
        "{ width w        |  460 | The width which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ height h       |  460 | The height which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ face_cascade f |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string directory(parser.get<string>("annotations"));
    //default initialisation
    Size scale(460,460);
    scale = Size(parser.get<int>("width"),parser.get<int>("height"));
    if (directory.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }
    string configfile_name(parser.get<string>("config"));
    if (configfile_name.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }
    string modelfile_name(parser.get<string>("model"));
    if (modelfile_name.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }
    string cascade_name(parser.get<string>("face_cascade"));
    if (cascade_name.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }
    //create a vector to store names of files in which annotations
    // and image names are found
    /*The format of the file containing annotations should be of following format
        /data/abc/abc.jpg
        123.45,345.65
        321.67,543.89

        The above format is similar to HELEN dataset which is used for training model 
     */
    string directory1 = directory+"*.txt";
    vector<String> filenames;
    //reading the files from the given directory
    glob(directory1,filenames);
    cout<<filenames.size();
    vector<String> filenames1,filenames2,filenames3;
    for(unsigned long i=0;i<=2000;i++){
        filenames1.push_back(filenames[i]);
    }
    vector<String> imagenames;
    //create object to get landmarks
    vector< vector<Point2f> > trainlandmarks;
    //create a pointer to call the base class
    //pass the face cascade xml file which you want to pass as a detector
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    Ptr<FacemarkKazemi> facemark= createFacemarkKazemi(face_cascade);
    //gets landmarks and corresponding image names in both the vectors
    facemark->getData(filenames1,trainlandmarks,imagenames);
    //vector to store images
    vector<Mat> trainimages;
    for(unsigned long i=0;i<imagenames.size();i++){
        string imgname = imagenames[i].substr(0, imagenames[i].size()-1);
        string img =string(directory+imgname + ".jpg");
        Mat src = imread(img);
        if(src.empty()){
            cout<<filenames1[i]<<endl;
            cerr<<string("Image"+img+"not found\n.Aborting...")<<endl;
            return 0;
        }
        trainimages.push_back(src);
    }
    cout<<"Got data"<<endl;
    //cout<<trainimages.size()<<endl;
    //cout<<trainlandmarks.size()<<endl;
    //Now scale data according to the size selected
    facemark->scaleData(trainlandmarks,trainimages,scale);
    //calculate mean shape
    cout<<"Scaled data"<<endl;
    vector<Rect> rectangles;
    facemark->calcMeanShape(trainlandmarks,trainimages,rectangles);
    cout<<trainimages.size()<<endl;
    cout<<trainlandmarks.size()<<endl;
    cout<<rectangles.size()<<endl;
    cout<<"Got mean shape"<<endl;
    /*//return (0);
    /*Now train data using training function which is yet to be built*/
    facemark->train(trainimages,trainlandmarks,rectangles,configfile_name,modelfile_name);
    cout<<"Training complete"<<endl;
    return 0; 
}
