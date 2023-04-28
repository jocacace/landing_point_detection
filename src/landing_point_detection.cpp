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
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TwistStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <mutex>
#include "geometry_msgs/Twist.h"
#include <geometry_msgs/PoseStamped.h>
#include "utils.h"

using namespace std;
using namespace Eigen;

std::mutex img_mutex;

// ****  REGRESSOR CLASS
class regressor3d {

    private:    
        Eigen::MatrixXd _X;
        Eigen::Vector3d _y; //COG
        Eigen::Matrix3d _cov;
        Eigen::Vector3d _versor;
        Eigen::JacobiSVD<Eigen::Matrix3d> _svd;
        int _n;        
        int _row_fill_i;

    public:
        regressor3d();
        void resize (int npoints);
        void fill_row(double &x,double &y, double &z);
        void compute_pca(void);
        void print_X(void){cout<<_X<<endl;}
        void print_cov(void){cout<<_cov<<endl;}
        Eigen::Vector3d get_COG(void){return _y;};
        Eigen::Vector3d get_versor(void){return _versor;};
 
};

regressor3d::regressor3d(){
    _n= 0;
    _row_fill_i =0;
    _cov = Eigen::Matrix3d::Zero();
    _y = Eigen::Vector3d::Zero();
    _versor = Eigen::Vector3d::Zero();
}

void regressor3d::resize(int npoints){
    _n = npoints;
    _X.resize(_n,3); 
    _X = Eigen::MatrixXd::Zero(_n,3);
    _row_fill_i = 0;  
    _cov = Eigen::Matrix3d::Zero(); 
    _y = Eigen::Vector3d::Zero();
    _versor = Eigen::Vector3d::Zero();
}

void regressor3d::fill_row(double &x,double &y, double &z){
    if(_row_fill_i <_n){
        _X.block(_row_fill_i,0,1,3) <<x,y,z;
        _row_fill_i++;
    }
}

void regressor3d::compute_pca(void)
{   
    _n = _row_fill_i; //just if matrix is filled with less elements than resized
    _X.resize(_n,3); 
    _y = _X.colwise().sum()/double(_n);
    //cout<<"sum is "<<_y<<endl; //
    _X = _X.rowwise() -_y.transpose();
    //cout<<_X<<endl; //
    _cov = (1.0f/double(_n))*(_X.transpose()*_X);
    //cout<<"covariance \n"<<_cov<<endl;
    //Eigen::JacobiSVD<Matrix3d, ComputeThinU | ComputeThinV> svd(_cov);
    //Eigen::JacobiSVD<Matrix3d, NoQRPreconditioner | ComputeThinU | ComputeThinV > svd(_cov);
    //Eigen::JacobiSVD<Eigen::Matrix3d> _svd(_cov,Eigen::ComputeFullU); // THIS IS THE ONE THAT WORK
    //cout << "Its singular values are:" << endl << svd.singularValues().transpose() << endl;
    //cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
    _svd.compute(_cov,Eigen::ComputeFullU);
    _versor = _svd.matrixU().block(0,0,3,1); //first col;
    _versor = _versor /_versor.norm();
    //cout<<"versor "<< _versor<<endl; //
}

// **** END REGRESSOR CLASS

class PIPE_INSPECTION {
    public:

        PIPE_INSPECTION();
        void extraction ();      
        void inspection( );
        void start_tracking_cb ( std_msgs::Bool data );        

        void run();
        void img_cb(const sensor_msgs::ImageConstPtr &rgb_msg, const sensor_msgs::ImageConstPtr &depth_msg);
        int pipeAxis_detect(cv::Mat depth_normalized, cv::Mat depthfloat,  cv::Mat depth, cv::Mat rgb, std::vector<cv::Point> &pipeAxis_pixls, cv::Point2f &dir_2d_axis, cv::Mat &mask_pipe, cv::Mat &output_img, geometry_msgs::TwistStamped & lp ,geometry_msgs::PoseStamped & landing_vect_pos);
       
    private:

        ros::NodeHandle _nh;
        message_filters::Subscriber<sensor_msgs::Image> _rgb_sub;
        message_filters::Subscriber<sensor_msgs::Image> _depth_sub;

        ros::Publisher _img_elaboration_output_pub;
        ros::Publisher _landing_point_pub;
        ros::Publisher _landing_vect_pub;
        //ros::Publisher _velocity_data_pub;
        //ros::ServiceServer _inspection_task_srv;
        tf2_ros::TransformBroadcaster tf_broadcast;
        tf::TransformListener tf_listener;

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
        
        float _inspection_distance = 0.8; 
        double _depth_scale_factor = 0.001; 
        double _depth_cutoff = 0.35;
        bool _land_on_nearest_point = true;
        std::string _rgb_topic; 
        std::string _depth_topic; 
        std::string _camera_info_topic;     
        std::string _depth_optical_frame;

        Eigen::Vector3d _pb_c;
        Eigen::Vector3d _e1_b_c;
        Eigen::Matrix3d _R_b_c;

        cv::Mat _rgb;
        cv::Mat _depth;
        //uint _n_pipe = 0;
        //int _change_dir = 0;
        int _n_not_axis = 0;
        cv::Mat _depth_normalized;

        ros::Subscriber _start_tracking_sub;
        bool _start_tracking = false;
        bool _first_img = false;
        cv::Mat *_cam_cameraMatrix, *_cam_distCo, *_cam_R, *_cam_P;

        bool _inspecting = false; //COMPROBAR
        bool _preempt_inspection;

        float _cx;
        float _cy;
        float _fx_inv; 
        float _fy_inv;

        vector< Eigen::Vector3d > _centroids;
        regressor3d regr;

        
};



PIPE_INSPECTION::PIPE_INSPECTION() {

    //reg = new regression();

    //_velocity_data_pub = _nh.advertise< geometry_msgs::Twist > ("/auto_control/twist", 1 );
    _img_elaboration_output_pub = _nh.advertise< sensor_msgs::Image > ("/pipe_line_extraction/elaboration/compressed", 1);
    _landing_point_pub = _nh.advertise< geometry_msgs::TwistStamped> ("/landing_point", 1);
    _landing_vect_pub = _nh.advertise< geometry_msgs::PoseStamped> ("/landing_vector", 1);
    if( !_nh.getParam("rgb_image", _rgb_topic) ) {
        _rgb_topic =  "/d400/color/image_raw";
    }

    if( !_nh.getParam("depth_image", _depth_topic) ) {
        _depth_topic =  "/d400/depth/image_raw";
    }
    
    if( !_nh.getParam("camera_info", _camera_info_topic) ) {
        _camera_info_topic =  "/d400/depth/camera_info";
    }

    if( !_nh.getParam("depth_optical_frame", _depth_optical_frame) ) {
        _depth_optical_frame =  "d400_depth_optical_frame";
    }

    if( !_nh.getParam("depth_scale_factor", _depth_scale_factor) ) {
        _depth_scale_factor =  0.001; // 1 in simulation (gazebo), 1000 for real RS camera
    }

    if( !_nh.getParam("depth_cutoff", _depth_cutoff) ) {
        _depth_cutoff =  0.35; // not considering pipe point too near camera to avoid "lambda like" skeleton at the end of fov.
    }

    if( !_nh.getParam("land_on_nearest_point", _land_on_nearest_point) ) {
        _land_on_nearest_point =  true; // if is true land on nearest point, else land on COM of the detected pipe
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
   
    //Get base link pose in camera frame ------------------------------------------------------
    _pb_c<<0.017, 0.089, -0.117; // default in case of tf failure
    _e1_b_c<<0.0, -0.939443, 0.342706;
    tf::StampedTransform tf_cam_base;
    tf::Quaternion q;
    bool found = false;
    int count = 0;
    while( !found && count++ < 10 ) {

        try{
            tf_listener.lookupTransform(_depth_optical_frame, "/base_link", ros::Time(0), tf_cam_base);
            _pb_c<<tf_cam_base.getOrigin().x(),tf_cam_base.getOrigin().y(),tf_cam_base.getOrigin().z();
            q = tf_cam_base.getRotation();
            _R_b_c = utilities::QuatToMat(Eigen::Vector4d(q.getW(),q.getX(),q.getY(),q.getZ()));
            _e1_b_c = _R_b_c*Eigen::Vector3d(1.0,0.0,0.0);
            found = true;
        }
        catch (tf::TransformException ex){ 
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
        
    }

    if( !found ) {
        ROS_ERROR("Non trovo la tf! addio!");
        exit(0);
    }
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
int PIPE_INSPECTION::pipeAxis_detect(cv::Mat depth_normalized, cv::Mat depthfloat, cv::Mat depth, cv::Mat rgb, std::vector<cv::Point> &pipeAxis_pixls,  cv::Point2f &dir_2d_axis, cv::Mat &mask_pipe, cv::Mat &output_img, geometry_msgs::TwistStamped & lp ,geometry_msgs::PoseStamped & landing_vect_pos) {
    
    double min;
    double max;
    int i = 0;

    _centroids.clear();


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
        
            float depth_pxl = depthfloat.at<float>(centroids.at<double>(i, 1), centroids.at<double>(i, 0) )*_depth_scale_factor; //Distance from the pipe
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
    
    // imshow("centroids", mask);
    // cv::waitKey(1);
    

    for(int r = 0; r < dst.rows; ++r) {
        for(int c = 0; c < dst.cols; ++c) {
            if (labelImage.at<int>(r, c) == id_big_cluster)
                dst.at<uchar>(r, c) = 255;
        }
    }

    // imshow("dst", dst);
    // cv::waitKey(1); 


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
                double z = depth.at<float>( (double)r, (double)c )*_depth_scale_factor; //Distance from the pipe
                if(z > _depth_cutoff){  // not considering pipe point too near camera to avoid "lambda like" skeleton at the end of fov.
                    x_SG.push_back((double)c);
                    y_SG.push_back((double)r);
                }

         
            }
        }
    }

    /// *** line regression
    cout<<"Npoints: "<<x_SG.size()<<endl;
    regr.resize(x_SG.size());
    for(int i=0; i<x_SG.size(); i++ ) {
        //pc[0] = (z) * ( (centroids.at<double>(i, 0) - _cx) * _fx_inv );
        //double z = depthfloat.at<float>( y_SG[i], x_SG[i] )*_depth_scale_factor; //Distance from the pipe
        double z = depth.at<float>( y_SG[i], x_SG[i] )*_depth_scale_factor; //Distance from the pipe
        double x = ( (z) * ( ( x_SG[i] - _cx) * _fx_inv ) );
        double y = ( (z) * ( ( y_SG[i] - _cy) * _fy_inv ) );
        regr.fill_row(x,y,z);

        if(z >= _depth_cutoff){
            circle(mask2, cv::Point(x_SG[i], y_SG[i]), 1, cv::Scalar(255,0,0), -1); // blue line: pipe skeleton
            
        }
        else{
            circle(mask2, cv::Point(x_SG[i], y_SG[i]), 1, cv::Scalar(0,0,255), -1); // red line: pipe skeleton
        }
    }



    if( x_SG.size() > 2 ) { //at least two point for a line

        //first and last pipe points
        circle(mask2, cv::Point(x_SG[0],y_SG[0]), 5, cv::Scalar(0,255,0), -1);
        circle(mask2, cv::Point(x_SG[x_SG.size()-1],y_SG[y_SG.size()-1]), 5, cv::Scalar(0,255,0), -1);

        regr.compute_pca();
        Eigen::Vector3d p_cog = regr.get_COG();
        Eigen::Vector3d v_land = regr.get_versor();
        Eigen::Vector3d p_land;
        
        /* choose versor signum according to drone heading*/
        cout<< "body_x in cam frame: "<<_e1_b_c<<endl;
        cout<< "scalar prod result: "<<_e1_b_c.dot(v_land)<<endl;
        if(_e1_b_c.dot(v_land)<0.0){
            v_land = -v_land;
        }
        
        cout<<"COM: "<<p_cog.transpose()<<endl;
        cout<<"versor: "<<v_land.transpose()<<endl;

        Eigen::Matrix3d R = utilities::versor2rotm(v_land );
        Eigen::Vector4d landing_quat = utilities::rot2quat(R);
        landing_quat = landing_quat/landing_quat.norm();

        if(_land_on_nearest_point){
            p_land = (v_land.dot(_pb_c -p_cog)/v_land.dot(v_land))*v_land +p_cog;
        }
        else{
            p_land =p_cog;
        }

        landing_vect_pos.pose.position.x = p_land[0];
        landing_vect_pos.pose.position.y = p_land[1];
        landing_vect_pos.pose.position.z = p_land[2];
        landing_vect_pos.pose.orientation.w = landing_quat[0];
        landing_vect_pos.pose.orientation.x = landing_quat[1];
        landing_vect_pos.pose.orientation.y = landing_quat[2];
        landing_vect_pos.pose.orientation.z = landing_quat[3];

        output_img = mask2;
        //imshow("skel", mask2);
        //cv::waitKey(1);
        return 1;
    }
    else { 
        output_img = mask2;
        lp.twist.linear.x = lp.twist.linear.y = lp.twist.linear.z = -1;
        lp.twist.linear.x = lp.twist.linear.y = lp.twist.linear.z = -1;
        landing_vect_pos.pose.position.x = 0;
        landing_vect_pos.pose.position.y = 0;
        landing_vect_pos.pose.position.z = 0;
        landing_vect_pos.pose.orientation.w = 1;
        landing_vect_pos.pose.orientation.x = 0;
        landing_vect_pos.pose.orientation.y = 0;
        landing_vect_pos.pose.orientation.z = 0;
        return -1;
    }
    
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
            ROS_WARN("Empty image");
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
        geometry_msgs::TwistStamped lp;
        geometry_msgs::PoseStamped land_vector;
        int retval = pipeAxis_detect(_depth_normalized, depthfloat, depth, rgb, pipeAxis_pxls, dir_axis_2d, mask_pipe, output_img, lp, land_vector);     
        //
        
        if (retval == -1) {
           //if (_start_tracking) 
           ROS_WARN("Not axis detected");
            // cv_ptr.image = mask2;
            // _img_elaboration_output_pub.publish( cv_ptr.toCompressedImageMsg() );

            _n_not_axis++;
            // cv::Mat mask2 = rgb.clone();
            // cv_ptr.image = mask2;
            // _img_elaboration_output_pub.publish( cv_ptr.toCompressedImageMsg() );      
            //continue;
        }
        _n_not_axis=0;       
        

        cout<< "ERROR STATUS = "<<retval<<endl<<endl;
        

        if(retval){
            cv_ptr.image = output_img;
            _img_elaboration_output_pub.publish( cv_ptr.toImageMsg() );
            //image plane approach (deprecated)
            lp.header.stamp = ros::Time::now();
            lp.header.frame_id = _depth_optical_frame;
            _landing_point_pub.publish( lp );

            // publish landing pose 
            land_vector.header.stamp = ros::Time::now();
            land_vector.header.frame_id = _depth_optical_frame;
            _landing_vect_pub.publish(land_vector);       
            // publish also tf
            geometry_msgs::TransformStamped transformStamped;
            transformStamped.header.stamp = ros::Time::now();
            transformStamped.header.frame_id = _depth_optical_frame;
            transformStamped.child_frame_id = "landing_frame";
            transformStamped.transform.translation.x = utilities::isnan(land_vector.pose.position.x) ? 0 : land_vector.pose.position.x;
            transformStamped.transform.translation.y = utilities::isnan(land_vector.pose.position.y) ? 0 : land_vector.pose.position.y;
            transformStamped.transform.translation.z = utilities::isnan(land_vector.pose.position.z) ? 0 : land_vector.pose.position.z;
            transformStamped.transform.rotation.x = utilities::isnan(land_vector.pose.orientation.x) ? 0 : land_vector.pose.orientation.x;
            transformStamped.transform.rotation.y = utilities::isnan(land_vector.pose.orientation.y) ? 0 : land_vector.pose.orientation.y;
            transformStamped.transform.rotation.z = utilities::isnan(land_vector.pose.orientation.z) ? 0 : land_vector.pose.orientation.z;
            transformStamped.transform.rotation.w = utilities::isnan(land_vector.pose.orientation.w) ? 1 : land_vector.pose.orientation.w;

            tf_broadcast.sendTransform(transformStamped);
        }
        rate.sleep();
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pipe_line_extraction");
    PIPE_INSPECTION inspection;
    return 0;
}

