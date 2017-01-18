#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void readme();

//AIRPLANE , BUS, WATERCRAFT,
static std::string arr[] = { "0","1","2"};


cv::Mat pcaAnalyse(cv::Mat &frameReference, cv::Mat &frameUnderTest, std::vector<cv::KeyPoint> &keypoints_1, cv::Mat &img_keypoints_1) {

	//GrayScale conversion
    cv::cvtColor(frameReference, frameUnderTest, CV_BGR2GRAY);
    //-- Draw keypoints
    
    cv::FeatureDetector * detector  = new cv::SIFT();
    // scale down image
    pyrDown( frameUnderTest, frameUnderTest, cv::Size( (int)(frameUnderTest.cols/2),(int) (frameUnderTest.rows/2)) );
    // detect features
    detector->detect( frameUnderTest, keypoints_1 );
    
    //std::cout << keypoints_1.size()<< " ";
    
    cv::DescriptorExtractor * extractor = new cv::SIFT();
    // extract keypoints in organised form
    extractor->compute(frameUnderTest,keypoints_1, img_keypoints_1);
    
    //PCA analysis
    
        
    int size = (img_keypoints_1.rows < 128 ) ? 128:img_keypoints_1.rows;
    cv::Mat data_pts = cv::Mat(size,128, CV_64FC1,cvScalar(0.0));
	// Creating PCA data buffer
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < data_pts.cols; ++j) {
            data_pts.at<double>(i, j) = img_keypoints_1.at<int>(i,j);
        }
    }  
    //Perform PCA analysis
    cv::PCA pca_analysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);
    
    //Store the eigenvalues and eigenvectors
    // eigen_vecs(128);
    cv::Mat eigen_vecs = cv::Mat(pca_analysis.eigenvectors.rows,pca_analysis.eigenvectors.cols, CV_8UC1);
    std::vector<double> eigen_val(2);
    
    int cnt=0,cntn=0 ;
    for (int i = 0; i < pca_analysis.eigenvectors.rows; ++i) {
        for (int j = 0; j < pca_analysis.eigenvectors.cols; ++j) {
            // eigen vecs values vary from 0 to 1
            eigen_vecs.at<uint8_t>(i,j)=static_cast<int>( pca_analysis.eigenvectors.at<double>(i, j)*255);
        }
    }
    return eigen_vecs;
}

/** @function main */
int main( int argc, char** argv )
{
    
    if( argc != 5)
    { readme(); return -1; }
    
    std::string sourcevid = argv[1];
    //// 1: <source_dir> 2 <train file> 3 <dump path> 4 <class label>
    //// Reading training video file
    std::string train ;
    train.append(argv[2]);
    train.append("train_");
    train.append(arr[atoi(argv[4])]);
    train.append(".txt");
    std::cout << train << std::endl;
    
    std::string dir;
    dir.append(argv[3]);
    std::string file;
    file.append(dir);
    file.append("train.txt");
    std::ifstream ftrain (train.c_str());
    std::ofstream fsave;
    fsave.open(file.c_str(), std::ios::out | std::ios::app );
    std::cout << sourcevid << std::endl;
    if (ftrain.is_open()) {
        std::string line;
        while ( getline (ftrain,line) ) {
            std::string videoin;
            int label;
            
            char c;
            int frameNum = -1;          // Frame counter
            ////Reading individual training video
            int index = line.find(" ");
            line = line.substr(0,index);
            line.append(".mp4");
            videoin.append(sourcevid);
            videoin.append(line);
            std::cout << line << '\n';
            label = atoi(argv[4]);
            cv::VideoCapture vidread(videoin);//, captUndTst(sourceCompareWith);
            
            if (!vidread.isOpened()) {
                std::cout  << "Could not open reference " << sourcevid << std::endl;
                return -1;
            }

            cv::Size refS = cv::Size((int) vidread.get(CV_CAP_PROP_FRAME_WIDTH),
                                     (int) vidread.get(CV_CAP_PROP_FRAME_HEIGHT));

            
            
            cv::Mat frameReference, frameUnderTest;

            for(;;) {//Show the image captured in the window and repeat
                vidread >> frameReference;

                
                if (frameReference.empty() ) {
                    std::cout << " < < <  Game over!  > > > ";
                    break;
                }

                ++frameNum;
                std::cout << "Frame: " << frameNum << "# ";
                std::vector<cv::KeyPoint> keypoints_1;
                cv::Mat img_keypoints_1;
                cv::Mat eigen_vecs = pcaAnalyse(frameReference, frameUnderTest, keypoints_1, img_keypoints_1);

                int idx1 = line.find("/");
                int idx2 = line.find(".");
                std::string save_img = line.substr(idx1+1,idx2-idx1-1);
                //char frame[5];
                //itoa(frameNum,frame,10);
                char frame[7];
                sprintf(frame,"%06d",frameNum);
                frame[6]='\0';
                save_img.append("_");
                save_img.append(frame);
                save_img.append(".jpg");
                std::cout << save_img << std::endl;

                std::string path;
                path.append(dir);
                path.append(save_img);
                fsave << save_img << " " << label << std::endl;
                std::cout << path << std::endl;
                imwrite(path,eigen_vecs);
                
                c = (char)cv::waitKey(1000/30);
               
            }
        }
        ftrain.close();
        fsave.close();
    }
        
    else std::cout << "Unable to open file";
    
    cv::waitKey(0);
    
    return 0;
}
    
    /** @function readme */
    void readme()
    { std::cout << " Usage: ./pca_video <source_dir> <train dataset file>  <dump path> <class label>" << std::endl; }
    
    /*
     g++ -I/usr/include/opencv -I/usr/include/opencv2 -L/usr/local/lib/ -g -o pca_video  pca_video.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching -lopencv_features2d -lopencv_nonfree
     */
    
