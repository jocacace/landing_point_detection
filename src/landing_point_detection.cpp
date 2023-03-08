/*
 *  Add copyright
 *
 *  
 *
 */

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf/transform_listener.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TwistStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <mutex>
#include "geometry_msgs/Twist.h"

using namespace std;
using namespace Eigen;

std::mutex img_mutex;

bool FROM_SIMU = true;


class regression {
   
   
 
public:
    // Store the coefficient/slope in
    // the best fitting line
    float coeff;
 
    // Store the constant term in
    // the best fitting line
    float constTerm;
 
    // Contains sum of product of
    // all (i-th x) and (i-th y)
    float sum_xy;
 
    // Contains sum of all (i-th x)
    float sum_x;
 
    // Contains sum of all (i-th y)
    float sum_y;
 
    // Contains sum of square of
    // all (i-th x)
    float sum_x_square;
 
    // Contains sum of square of
    // all (i-th y)
    float sum_y_square;

    // Constructor to provide the default
    // values to all the terms in the
    // object of class regression
    // Dynamic array which is going
    // to contain all (i-th x)
    vector<float> x;
 
    // Dynamic array which is going
    // to contain all (i-th y)
    vector<float> y;
 

    regression()
    {
        coeff = 0;
        constTerm = 0;
        sum_y = 0;
        sum_y_square = 0;
        sum_x_square = 0;
        sum_x = 0;
        sum_xy = 0;
    }
 
    // Function that calculate the coefficient/
    // slope of the best fitting line
    void calculateCoefficient()
    {
        float N = x.size();
        float numerator
            = (N * sum_xy - sum_x * sum_y);
        float denominator
            = (N * sum_x_square - sum_x * sum_x);
        coeff = numerator / denominator;
    }
 
    // Member function that will calculate
    // the constant term of the best
    // fitting line
    void calculateConstantTerm()
    {
        float N = x.size();
        float numerator
            = (sum_y * sum_x_square - sum_x * sum_xy);
        float denominator
            = (N * sum_x_square - sum_x * sum_x);
        constTerm = numerator / denominator;
    }
 
    // Function that return the number
    // of entries (xi, yi) in the data set
    int sizeOfData()
    {
        return x.size();
    }
 
    // Function that return the coefficient/
    // slope of the best fitting line
    float coefficient()
    {
        if (coeff == 0)
            calculateCoefficient();
        return coeff;
    }
 
    // Function that return the constant
    // term of the best fitting line
    float constant()
    {
        if (constTerm == 0)
            calculateConstantTerm();
        return constTerm;
    }
 
    // Function that print the best
    // fitting line
    void PrintBestFittingLine()
    {
        if (coeff == 0 && constTerm == 0) {
            calculateCoefficient();
            calculateConstantTerm();
        }
        cout << "The best fitting line is y = "
             << coeff << "x + " << constTerm << endl;

        cout << "Orientation: " << atan( coeff ) << endl;
    }
 
    // Function to take input from the dataset
    void takeInput(int n)
    {
        for (int i = 0; i < n; i++) {
            // In a csv file all the values of
            // xi and yi are separated by commas
            char comma;
            float xi;
            float yi;
            cin >> xi >> comma >> yi;
            sum_xy += xi * yi;
            sum_x += xi;
            sum_y += yi;
            sum_x_square += xi * xi;
            sum_y_square += yi * yi;
            x.push_back(xi);
            y.push_back(yi);
        }
    }
 
    // Function to show the data set
    void showData()
    {
        for (int i = 0; i < 62; i++) {
            printf("_");
        }
        printf("\n\n");
        printf("|%15s%5s %15s%5s%20s\n",
               "X", "", "Y", "", "|");
 
        for (int i = 0; i < x.size(); i++) {
            printf("|%20f %20f%20s\n",
                   x[i], y[i], "|");
        }
 
        for (int i = 0; i < 62; i++) {
            printf("_");
        }
        printf("\n");
    }
 
    // Function to predict the value
    // corresponding to some input
    float predict(float x)
    {
        return coeff * x + constTerm;
    }
 
    // Function that returns overall
    // sum of square of errors
    float errorSquare()
    {
        float ans = 0;
        for (int i = 0;
             i < x.size(); i++) {
            ans += ((predict(x[i]) - y[i])
                    * (predict(x[i]) - y[i]));
        }
        return ans;
    }
 
    // Functions that return the error
    // i.e the difference between the
    // actual value and value predicted
    // by our model
    float errorIn(float num)
    {
        for (int i = 0;
             i < x.size(); i++) {
            if (num == x[i]) {
                return (y[i] - predict(x[i]));
            }
        }
        return 0;
    }
};


class PIPE_INSPECTION {
    public:

        PIPE_INSPECTION();
        void extraction ();      
        void inspection( );
        void start_tracking_cb ( std_msgs::Bool data );        

        void run();
        void img_cb(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg);
        int pipeAxis_detect(cv::Mat depth_normalized, cv::Mat depthfloat,  cv::Mat depth, cv::Mat rgb, std::vector<cv::Point> &pipeAxis_pixls, cv::Point2f &dir_2d_axis, cv::Mat &mask_pipe, cv::Mat &output_img, geometry_msgs::Twist & lp );
       
    private:

        ros::NodeHandle _nh;
        message_filters::Subscriber<sensor_msgs::Image> _rgb_sub;
        message_filters::Subscriber<sensor_msgs::Image> _depth_sub;

        ros::Publisher _img_elaboration_output_pub;
        ros::Publisher _landing_point_pub;

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                                sensor_msgs::Image> MySyncPolicy;
        
        float _inspection_distance = 0.8; 
        float _depth_scale = 0.001; 
        
        std::string _rgb_topic; 
        std::string _depth_topic; 
        std::string _camera_info_topic;         
        ros::Publisher _velocity_data_pub;

        //ros::ServiceServer _inspection_task_srv;

        //Subscription info
        cv::Mat _rgb;
        cv::Mat _depth;

        uint _n_pipe = 0;
        int _change_dir = 0;
        int _n_not_axis = 0;
        cv::Mat _depth_normalized;

        ros::Subscriber _start_tracking_sub;
        bool _start_tracking = false;
        bool _first_img = false;
        cv::Mat *_cam_cameraMatrix, *_cam_distCo, *_cam_R, *_cam_P;

        double _kp_x, _kp_y, _kp_z;
        double _kd_x, _kd_y, _kd_z;
        double _ki_x, _ki_y, _ki_z;

        bool _inspecting = false; //COMPROBAR
        bool _preempt_inspection;

        float _cx;
        float _cy;
        float _fx_inv; 
        float _fy_inv;

        vector< Eigen::Vector3d > _centroids;
        regression *reg;

        
};



PIPE_INSPECTION::PIPE_INSPECTION() {

    reg = new regression();

    _velocity_data_pub = _nh.advertise< geometry_msgs::Twist > ("/auto_control/twist", 1 );
    _img_elaboration_output_pub = _nh.advertise< sensor_msgs::Image > ("/pipe_line_extraction/elaboration/compressed", 1);
    _landing_point_pub = _nh.advertise< geometry_msgs::Twist> ("/landing_point", 1);
    if( !_nh.getParam("rgb_image", _rgb_topic) ) {
        _rgb_topic =  "/camera/color/image_raw";
    }

    if( !_nh.getParam("depth_image", _depth_topic) ) {
        _depth_topic =  "/camera/depth/image_raw";
    }
    
    if( !_nh.getParam("camera_info", _camera_info_topic) ) {
        _camera_info_topic =  "/camera/depth/camera_info";
    }
        
    if( !_nh.getParam("kp_x", _kp_x) ) {
        _kp_x =  0.0;
    }
    if( !_nh.getParam("kp_y", _kp_y) ) {
        _kp_y =  0.0;
    }
    if( !_nh.getParam("kp_z", _kp_z) ) {
        _kp_z =  0.0;
    }


    if( !_nh.getParam("kd_x", _kd_x) ) {
        _kd_x =  0.0;
    }
    if( !_nh.getParam("kd_y", _kd_y) ) {
        _kd_y =  0.0;
    }
    if( !_nh.getParam("kd_z", _kd_z) ) {
        _kd_z =  0.0;
    }


    if( !_nh.getParam("ki_x", _ki_x) ) {
        _ki_x =  0.0;
    }
    if( !_nh.getParam("ki_y", _ki_y) ) {
        _ki_y =  0.0;
    }
    if( !_nh.getParam("ki_z", _ki_z) ) {
        _ki_z =  0.0;
    }


    //Get camera info---------------------------------------------------------------------------------------------
    _cam_cameraMatrix = new cv::Mat(3, 3, CV_64FC1);
    _cam_distCo = new cv::Mat(1, 5, CV_64FC1);
    _cam_R = new cv::Mat(3, 3, CV_64FC1);
    _cam_P = new cv::Mat(3, 4, CV_64FC1);
    sensor_msgs::CameraInfo camera_info;
    boost::shared_ptr<sensor_msgs::CameraInfo const> sharedCamera_info;
    sharedCamera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(_camera_info_topic,_nh);
    if(sharedCamera_info != NULL){
        camera_info = *sharedCamera_info;


        //---K
        for(int i=0; i<3;i++) {
            for(int j=0; j<3; j++) {
                _cam_cameraMatrix->at<double>(i,j) = camera_info.K[3*i+j];
            }
        }
        
        //---D
        if( camera_info.D.size() >= 5 ) {
            for(int i=0; i<5;i++) {
                _cam_distCo->at<double>(0,i) = camera_info.D[i];
            }
        }
        
        //---R
        for(int i=0; i<3;i++) {
            for(int j=0; j<3; j++) {
                _cam_R->at<double>(i,j) = camera_info.R[3*i+j];
            }
        }
        
        //---P
        for(int i=0; i<3;i++) {
            for(int j=0; j<4; j++) {
                _cam_P->at<double>(i,j) = camera_info.P[4*i+j];
            }
        }
    }

    _cx = _cam_cameraMatrix->at<double>(0,2);
    _cy = _cam_cameraMatrix->at<double>(1,2);
    _fx_inv = 1.0 / _cam_cameraMatrix->at<double>(0,0);
    _fy_inv = 1.0 / _cam_cameraMatrix->at<double>(1,1);
   
    _start_tracking_sub = _nh.subscribe("/joy_control/start_tracking", 1, &PIPE_INSPECTION::start_tracking_cb, this);
    _rgb_sub.subscribe(_nh, _rgb_topic, 1);
    _depth_sub.subscribe(_nh, _depth_topic, 1);


    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), _rgb_sub, _depth_sub);
    sync.registerCallback(boost::bind(&PIPE_INSPECTION::img_cb, this, _1, _2)); //cb_pid // cb3

    boost::thread inspection_t( &PIPE_INSPECTION::inspection, this);

    ros::spin();
    
}


void PIPE_INSPECTION::start_tracking_cb ( std_msgs::Bool b) {
    _start_tracking = b.data;
} // At the start, the UAV must be bring ahead the pipe 
  // This callback is used to require the start of the tracking algorithm 

bool polyfit( const std::vector<double> &t,
              const std::vector<double> &v,
              std::vector<double> &coeff,
              int order) {

    // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
    Eigen::MatrixXd T(t.size(), order + 1);
    Eigen::VectorXd V = Eigen::VectorXd::Map(&v.front(), v.size());
    Eigen::VectorXd result;

    if( (t.size() != v.size() ) || ( t.size() < (order + 1)   ) ) {
        ROS_WARN("Assert failed");
        return false;
    }

    // Populate the matrix
    for(size_t i = 0 ; i < t.size(); ++i) {
        for(int j = 0; j < order + 1; ++j)
        {
            T(i, j) = pow(t.at(i), j);
        }
    }
    
    result  = T.householderQr().solve(V);
    coeff.resize(order+1);
    for (int k = 0; k < order+1; k++) {
        coeff[k] = result[k];
    }

    return true;
}

//Function implementing the pipe axis detection
int PIPE_INSPECTION::pipeAxis_detect(cv::Mat depth_normalized, cv::Mat depthfloat, cv::Mat depth, cv::Mat rgb, std::vector<cv::Point> &pipeAxis_pixls,  cv::Point2f &dir_2d_axis, cv::Mat &mask_pipe, cv::Mat &output_img, geometry_msgs::Twist & lp ) {
    
    double min;
    double max;
    int i = 0;

    _centroids.clear();
    reg->x.clear();
    reg->y.clear();
    reg->coeff = 0;
    reg->constTerm = 0;
    reg->sum_y = 0;
    reg->sum_y_square = 0;
    reg->sum_x_square = 0;
    reg->sum_x = 0;
    reg->sum_xy = 0;

    for(int r=0;r<depth_normalized.rows;r++) {
        for(int c=0; c<depth_normalized.cols;c++) {
            if (depth_normalized.at<uchar>(r,c)==0) {
                depth_normalized.at<uchar>(r,c)=255;
            }
        }
    }

    cv::medianBlur(depth_normalized, depth_normalized, 15);
    cv::Mat mask = cv::Mat::zeros(depth_normalized.size(),CV_8UC1);
    cv::Mat depth_normalized_aux = depth_normalized.clone();

    int n_pxls=0;
    while(n_pxls < depth_normalized.cols*depth_normalized.rows/10) {
        n_pxls=0;
        cv::minMaxIdx(depth_normalized_aux, &min, &max);
        for(int r=0;r<depth_normalized_aux.rows;r++) {
            for(int c=0; c<depth_normalized_aux.cols;c++) {
                if (depth_normalized_aux.at<uchar>(r,c)<min+3*min/4) {
                    depth_normalized_aux.at<uchar>(r,c) = 255;
                    mask.at<uchar>(r,c)=255;
                    n_pxls++;
                }
            }
        }
    }

    //Open operation
    //cv::morphologyEx(mask,mask,cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT,cv::Size(80,80)));
    cv::morphologyEx(mask,mask,cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_DILATE,cv::Size(5,5)));
    //Get bigger cluster
    cv::Mat labelImage(mask.size(), CV_32S);
    //int nLabels = connectedComponents(mask, labelImage, 8);
    cv::Mat stats;
    cv::Mat centroids;
    
    int nLabels = connectedComponentsWithStats(mask, labelImage, stats, centroids, 8);  //0 is the background

    for(int i=0; i<nLabels; i++ ) {
        Eigen::Vector3d pc;
       
        if ( i==0 ) {
            pc << 0.0, 0.0, 0.0;
            _centroids.push_back ( pc );
        }
        else {
            circle(mask, cv::Point(centroids.at<double>(i, 0), centroids.at<double>(i, 1)), 4, cv::Scalar(0,255,0), -1);
        
            float depth_pxl = depthfloat.at<float>(centroids.at<double>(i, 1), centroids.at<double>(i, 0) ); //Distance from the pipe
            pc[0] = (depth_pxl) * ( (centroids.at<double>(i, 0) - _cx) * _fx_inv );
            pc[1] = (depth_pxl) * ( (centroids.at<double>(i, 1) - _cy) * _fy_inv );
            pc[2] = depth_pxl;
            _centroids.push_back ( pc );            
        }
    }

    std::vector<int>labels_area(nLabels, 0);
    cv::Mat dst = cv::Mat::zeros(mask.size(), CV_8UC1);
    for(int r = 0; r < dst.rows; ++r) {
        for(int c = 0; c < dst.cols; ++c) {
            int label = labelImage.at<int>(r, c);
            if (mask.at<uchar>(r,c) != 0)
                labels_area[label]++;
        }
    }

    int id_big_cluster;
    int max_pixels=0;
    for (int i=0; i<nLabels; i++) {
        if(labels_area[i]>max_pixels) {
            max_pixels = labels_area[i];
            id_big_cluster = i;
        }
    }

    float closer_centroid_dist = 10000;
    for( int i=1; i<nLabels; i++ ) {
        if( _centroids[i].norm() < closer_centroid_dist) {
            closer_centroid_dist = _centroids[i].norm();
            id_big_cluster = i;
        }
    }
    circle(mask, cv::Point(centroids.at<double>(id_big_cluster, 0), centroids.at<double>(id_big_cluster, 1)), 12, cv::Scalar(0,255,0), -1);
    
    //imshow("centroids", mask);
    //cv::waitKey(1);
    

    for(int r = 0; r < dst.rows; ++r) {
        for(int c = 0; c < dst.cols; ++c) {
            if (labelImage.at<int>(r, c) == id_big_cluster)
                dst.at<uchar>(r, c) = 255;
        }
    }

    //imshow("dst", dst);
    //cv::waitKey(1); 


    mask = dst.clone();
    cv::GaussianBlur(mask,mask,cv::Size(11,11),2.0);
    mask_pipe = mask.clone();
    cv::Mat mask2;
    cvtColor(mask, mask2, CV_GRAY2BGR);

    cv::Mat dist;
    distanceTransform(mask, dist, cv::DIST_L1, 3);
    normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

    cv::Mat skeleton;
    cv::Mat skel(mask.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
 
    bool done;
    do {
          cv::morphologyEx(mask, temp, cv::MORPH_OPEN, element);
          cv::bitwise_not(temp, temp);
          cv::bitwise_and(mask, temp, temp);
          cv::bitwise_or(skel, temp, skel);
          cv::erode(mask, mask, element);
          double max;
          cv::minMaxLoc(mask, 0, &max);
          done = (max == 0);
    } while (!done);


    cv::morphologyEx(skel, skel, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_CROSS, cv::Size(1,1)));
    threshold( skel, skel, 250, 255, cv::THRESH_BINARY );
    skeleton = skel;

    std::vector<double> x_SG, y_SG;
    std::vector<double> x_SG_, y_SG_;
    std::vector<cv::Point> puntos;

    for(int r = 0; r < skeleton.rows; ++r) {
        for(int c = 0; c < skeleton.cols; ++c) {
            if (skeleton.at<uchar>(r, c) == 255) {
                if (r < skeleton.rows/10 || r > skeleton.rows - skeleton.rows/10) {
                    skeleton.at<uchar>(r, c) = 0;
                    continue;
                }
                if (c < skeleton.cols/10 || c > skeleton.cols - skeleton.cols/10) {
                    skeleton.at<uchar>(r, c) = 0;
                    continue;
                }
                if (dist.at<float>(r, c) < 0.5) {
                    continue;
                }

                std::pair<int, int> punto(c, r);
                puntos.push_back(cv::Point(c,r));
                x_SG.push_back((double)c);
                y_SG.push_back((double)r);

         
            }
        }
    }

    for(int i=0; i<x_SG.size(); i++ ) {


        float depth_pxl = depthfloat.at<float>( y_SG[i], x_SG[i] ); //Distance from the pipe
        //pc[0] = (depth_pxl) * ( (centroids.at<double>(i, 0) - _cx) * _fx_inv );

        float x = ( (depth_pxl) * ( ( x_SG[i] - _cx) * _fx_inv ) );
        float y = ( (depth_pxl) * ( ( y_SG[i] - _cy) * _fy_inv ) );


        reg->sum_xy += x*y; //0.0; //xi * yi;
        reg->sum_x += x; //xi;
        reg->sum_y += y; //yi;
        reg->sum_x_square += x*x; //xi * xi;
        reg->sum_y_square += y*y; //yi * yi;
    
        reg->x.push_back( x );        
        reg->y.push_back( y );

    
        circle(mask2, cv::Point(x_SG[i], y_SG[i]), 1, cv::Scalar(255,0,0), -1);


    }



    if( x_SG.size() > 0 ) {
        reg->PrintBestFittingLine();

        //landing point: the mean one
        float mX = (x_SG[ x_SG.size()-1] + x_SG[0]) / 2.0;
        float mY = (y_SG[ x_SG.size()-1] + y_SG[0]) / 2.0;

        circle(mask2, cv::Point(mX, mY), 5, cv::Scalar(0,255,0), -1);

        //circle(mask2, cv::Point(x_SG[0],y_SG[0]), 5, cv::Scalar(0,255,0), -1);
        //circle(mask2, cv::Point(x_SG[x_SG.size()-1],y_SG[y_SG.size()-1]), 5, cv::Scalar(0,255,0), -1);


        float depth_cpxl = depth.at<float>(mY, mX ); //Distance from the pipe
        //depth_cpxl *= 100;
        lp.linear.x = (depth_cpxl) * ( (mX - _cx) * _fx_inv );
        lp.linear.y = (depth_cpxl) * ( (mY - _cy) * _fy_inv );
        lp.linear.z = depth_cpxl;

        output_img = mask2;
        //imshow("skel", mask2);
        //cv::waitKey(1);

        //return: landing point poistion: x,y,z
        //        rotation

    }
    else { 
        output_img = mask2;
        lp.linear.x = lp.linear.y = lp.linear.z = -1;
        lp.angular.x = lp.angular.y = lp.angular.z = -1;
    }
    return 1;
}

//Work on this function
void PIPE_INSPECTION::img_cb(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg) {

    cv_bridge::CvImagePtr cv_ptr_rgb, cv_ptr_depth;
    try {
       cv_ptr_rgb = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::TYPE_8UC3);
       cv_ptr_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }

    img_mutex.lock();
    _rgb = cv_ptr_rgb->image;
    _depth = cv_ptr_depth->image;

    //cout << _depth.at<float>(240, 320 )   << endl;    
    _first_img = true;
    img_mutex.unlock();
}



void PIPE_INSPECTION::inspection()  {

    while(!_first_img  ) {
        usleep(0.1*1e6);
        cout << "wait first image" << endl;
    } 
    
    ros::Rate rate(10);
    
    double min;
    double max;
    cv::Mat rgb;
    cv::Mat depth;
    cv::Mat mask2;

    cv_bridge::CvImage cv_ptr;

    geometry_msgs::Twist cmd_vel;

    float cx = _cam_cameraMatrix->at<double>(0,2);
    float cy = _cam_cameraMatrix->at<double>(1,2);
    float fx_inv = 1.0 / _cam_cameraMatrix->at<double>(0,0);
    float fy_inv = 1.0 / _cam_cameraMatrix->at<double>(1,1);
    double dir[3];

    Eigen::Vector3d e_p;
    Eigen::Vector3d e_pp;
    Eigen::Vector3d de_p;
    Eigen::Vector3d ie_p;
    Eigen::Vector3d ctrl_vel;

    e_p << 0.0, 0.0, 0.0;
    e_pp << 0.0, 0.0, 0.0;
    de_p << 0.0, 0.0, 0.0;
    ie_p << 0.0, 0.0, 0.0;

    cv::Mat depthfloat;
    cv_ptr.encoding = "bgr8";

    std::vector<cv::Point> pipeAxis_pxls;
    cv::Point2f dir_axis_2d;
    cv::Mat mask_pipe;

    bool inspection_done = false;
    while( ros::ok()  ) {

        //---Init
        pipeAxis_pxls.clear();   
        
        cv_ptr.header.stamp = ros::Time::now();
        img_mutex.lock();
        if( _depth.empty() || _rgb.empty() ) {
            img_mutex.unlock(); 
            continue;
        }      
        _depth.copyTo(depth);
        _rgb.copyTo(rgb);
        img_mutex.unlock(); //Local images
        depth.convertTo(depthfloat, cv::DataType<float>::type);
        //---

        cv::minMaxIdx(depth, &min, &max);
        cv::convertScaleAbs(depth, _depth_normalized, 255 / max);        
        
        mask2 = rgb.clone();


        //imshow( "depth_normalized", _depth_normalized);
        //cv::waitKey(1);
        cv::Mat output_img;
        geometry_msgs::Twist lp;
        int error = pipeAxis_detect(_depth_normalized, depthfloat, depth, rgb, pipeAxis_pxls, dir_axis_2d, mask_pipe, output_img, lp);     
        //
        /*
        if (error == -1) {
           if (_start_tracking) ROS_WARN("Not axis detected");
            cv_ptr.image = mask2;
            _img_elaboration_output_pub.publish( cv_ptr.toCompressedImageMsg() );

            _n_not_axis++;
            cv::Mat mask2 = rgb.clone();
            cv_ptr.image = mask2;
            _img_elaboration_output_pub.publish( cv_ptr.toCompressedImageMsg() );      
            continue;
        }
        _n_not_axis=0;

        
        */
        cv_ptr.image = output_img;
        _img_elaboration_output_pub.publish( cv_ptr.toImageMsg() );
        _landing_point_pub.publish( lp );
        
        rate.sleep();
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pipe_line_extraction");
    PIPE_INSPECTION inspection;
    return 0;
}

