#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <shape_msgs/SolidPrimitive.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2/convert.h>
#include <tf2_ros/transform_listener.h>

#include <navigator_msgs/Object.h>
#include <navigator_msgs/Objects.h>
#include <navigator_msgs/BuoyArray.h>
#include <navigator_msgs/Buoy.h>
#include <uf_common/PoseTwistStamped.h>
#include <uf_common/MoveToAction.h>
#include <actionlib/server/simple_action_server.h>

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "OccupancyGrid.h"
#include "ConnectedComponents.h"
#include "AStar.h"
#include "objects.h"
#include "bounding_boxes.h"

using namespace std;

const double MAP_SIZE_METERS = 1500.3;
const double ROI_SIZE_METERS = 201.3;
const double VOXEL_SIZE_METERS = 0.30;
const int MIN_HITS_FOR_OCCUPANCY = 50; //20
const int MAX_HITS_IN_CELL = 500; //500
const double MAXIMUM_Z_HEIGHT = 8;

OccupancyGrid ogrid(MAP_SIZE_METERS,ROI_SIZE_METERS,VOXEL_SIZE_METERS);
nav_msgs::OccupancyGrid rosGrid;
ros::Publisher pubGrid, pubMarkers, pubBuoys, pubObjPC, pubPC;

visualization_msgs::MarkerArray markers;
visualization_msgs::Marker m;
ObjectTracker object_tracker;

geometry_msgs::Point waypoint_ogrid;
geometry_msgs::Pose boatPose_enu;
geometry_msgs::Twist boatTwist_enu;
uf_common::PoseTwistStamped waypoint_enu, carrot_enu;

float LLA_BOUNDARY_X1 = -30, LLA_BOUNDARY_Y1 = 50;
float LLA_BOUNDARY_X2 = -30, LLA_BOUNDARY_Y2 = -20;
float LLA_BOUNDARY_X3 = 35, LLA_BOUNDARY_Y3 = -20;
float LLA_BOUNDARY_X4 = 35, LLA_BOUNDARY_Y4 = 50;

union fc{
    unsigned char c[4];
    float f;
};

sensor_msgs::PointCloud2 gp32_helper(objectMessage& object, bool debug_color = false){
    /* Converts a vector of geometry_msgs/Point32 to sensor_msgs/PointCloud2 */
    /* TODO: Possibly convert this to pcl_ros PointCloud<T> message to save time */

    sensor_msgs::PointCloud2 _cloud;
    std::vector<float> _data;
    std::vector<unsigned char> _dc;

    _cloud.is_dense = false;
    _cloud.header.stamp = ros::Time::now();
    _cloud.header.frame_id = "enu";          /* TODO: Do not hardcode this */
    _cloud.height = 1;                       /* Unorganized clouds have no height */
    _cloud.width = object.beams.size();
    _cloud.point_step = 12;
    _cloud.row_step = 1;

    /* TODO: Add intensity information */
    std::vector<sensor_msgs::PointField> _p;
    sensor_msgs::PointField _px, _py, _pz;
    fc _fc;

    _px.name = 'x';     _px.offset = 0;     _px.datatype = sensor_msgs::PointField::FLOAT32;    _px.count = 1;
    _py.name = 'y';     _py.offset = 4;     _py.datatype = sensor_msgs::PointField::FLOAT32;    _py.count = 1;
    _pz.name = 'z';     _pz.offset = 8;     _pz.datatype = sensor_msgs::PointField::FLOAT32;    _pz.count = 1;
    _p.push_back(_px);  _p.push_back(_py);  _p.push_back(_pz);

    _cloud.fields = _p;

    for (std::vector<geometry_msgs::Point32>::iterator it = object.beams.begin(); it != object.beams.end(); ++it){
        _data.push_back((*it).x);
        _data.push_back((*it).y);
        _data.push_back((*it).z);
    }

    for(std::vector<float>::iterator f = _data.begin(); f != _data.end(); ++f){
        unsigned char *_d = reinterpret_cast<unsigned char *>(&*f);
        for(std::size_t i = 0; i != sizeof(float); ++i){
            _fc.c[i] = _d[i];
            _dc.push_back(_d[i]);
        }
    }
    std::cout << "Object Size: " << _data.size() << std::endl;

    std::cout << "Number of Points: " << _data.size() << std::endl;
    std::cout << "Size of DC: " << _dc.size() << std::endl;

    _cloud.data = _dc;
    return _cloud;
}

void actionExecute(const uf_common::MoveToGoalConstPtr& goal)
{
    //Grab new goal from actionserver
    ROS_INFO("LIDAR: Following new goal from action server!");
    waypoint_enu.posetwist = goal->posetwist;
}

void cb_velodyne(const sensor_msgs::PointCloud2ConstPtr &pcloud)
{
    ROS_INFO("**********************************************************");
    ROS_INFO("LIDAR | cb_velodyne...");

    //Measure elapsed time for function
    ros::Time timer = ros::Time::now();

    //Use ROS transform listener to grad up-to-date transforms between reference frames
    static tf2_ros::Buffer tfBuffer;
    static tf2_ros::TransformListener tfListener(tfBuffer);
    geometry_msgs::TransformStamped T_enu_velodyne_ros;
    try {
        T_enu_velodyne_ros = tfBuffer.lookupTransform("enu", "velodyne",ros::Time(0)); //change time to pcloud header? pcloud->header.stamp
    } catch (tf2::TransformException &ex) {
      ROS_ERROR("%s",ex.what());
      return;
    }

    //Convert ROS transform to eigen transform
    Eigen::Affine3d T_enu_velodyne(Eigen::Affine3d::Identity());
    geometry_msgs::Vector3  lidarpos =  T_enu_velodyne_ros.transform.translation;
    geometry_msgs::Quaternion quat = T_enu_velodyne_ros.transform.rotation;
    T_enu_velodyne.translate(Eigen::Vector3d(lidarpos.x,lidarpos.y,lidarpos.z));
    T_enu_velodyne.rotate(Eigen::Quaterniond(quat.w,quat.x,quat.y,quat.z));
    ROS_INFO_STREAM("LIDAR | Velodyne enu: " << lidarpos.x << "," << lidarpos.y << "," << lidarpos.z);

    //Update occupancy grid
    ogrid.setLidarPosition(lidarpos);
    ogrid.updatePointsAsCloud(pcloud,T_enu_velodyne,MAX_HITS_IN_CELL);
    ogrid.createBinaryROI(MIN_HITS_FOR_OCCUPANCY,MAXIMUM_Z_HEIGHT);

    //Inflate ogrid before detecting objects and calling AStar
    ogrid.inflateBinary(1);

    //Detect objects
    std::vector<objectMessage> objects;
    std::vector< std::vector<int> > cc = ConnectedComponents(ogrid, objects);

    //Publish rosgrid
    rosGrid.header.seq = 0;
    rosGrid.info.resolution = VOXEL_SIZE_METERS;
    rosGrid.header.frame_id = "enu";
    rosGrid.header.stamp = ros::Time::now();
    rosGrid.info.map_load_time = ros::Time::now();
    rosGrid.info.width = ogrid.ROI_SIZE;
    rosGrid.info.height = ogrid.ROI_SIZE;
    rosGrid.info.origin.position.x = ogrid.lidarPos.x + ogrid.ROItoMeters(0);
    rosGrid.info.origin.position.y = ogrid.lidarPos.y + ogrid.ROItoMeters(0);
    rosGrid.info.origin.position.z = ogrid.lidarPos.z;
    rosGrid.data = ogrid.ogridMap;
    pubGrid.publish(rosGrid);

    //Publish markers
    geometry_msgs::Point p;
    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker m;
    m.header.stamp = ros::Time::now();
    m.header.seq = 0;
    m.header.frame_id = "enu";

    //Erase old markers
    m.id = 1000;
    m.type = 0;
    m.action = 3;
    markers.markers.push_back(m);

    //Course Outline - change to real values or pull from service/topic
    m.id = 1001;
    m.type = visualization_msgs::Marker::LINE_STRIP;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.5;

    p.x = LLA_BOUNDARY_X1; p.y = LLA_BOUNDARY_Y1; p.z = lidarpos.z;
    m.points.push_back(p);
    p.x = LLA_BOUNDARY_X2; p.y = LLA_BOUNDARY_Y2; p.z = lidarpos.z;
    m.points.push_back(p);
    p.x = LLA_BOUNDARY_X3; p.y = LLA_BOUNDARY_Y3; p.z = lidarpos.z;
    m.points.push_back(p);
    p.x = LLA_BOUNDARY_X4; p.y = LLA_BOUNDARY_Y4; p.z = lidarpos.z;
    m.points.push_back(p);
    p.x = LLA_BOUNDARY_X1; p.y = LLA_BOUNDARY_Y1; p.z = lidarpos.z;
    m.points.push_back(p);
    m.color.a = 0.6; m.color.r = 1; m.color.g = 1; m.color.b = 1;
    markers.markers.push_back(m);

    /* Publish buoys */
    navigator_msgs::BuoyArray allBuoys;
    navigator_msgs::Buoy buoy;

    navigator_msgs::Object obj;
    navigator_msgs::Objects objs;
    geometry_msgs::Point32 p32;

    buoy.header.seq = 0;
    buoy.header.frame_id = "enu";
    buoy.header.stamp = ros::Time::now();

    auto object_permanence = object_tracker.add_objects(objects);
    std::vector<objectMessage> small_objects = BoundingBox::get_accurate_objects(pcloud, object_permanence, T_enu_velodyne);
    int max_id = 0;

    std::vector<navigator_msgs::Object> _o;

    for (auto o : object_permanence) {
        /* Preliminary segmentation of Velodyne Point Cloud */
        /* TODO: Change stamp to reflect recieve time for message */
        obj.header.stamp = ros::Time::now();
        obj.header.frame_id = "enu";

        obj.position = o.position;          /* Position in ENU frame in which the 'object' is found */
        obj.scale = o.scale;                /* Region that is bounding the detected 'object' */
        obj.cloud = gp32_helper(o, false);  /* sensor_msgs/PointCloud2 formatted cloud containing object points */
        obj.volume.data = float(obj.scale.x * obj.scale.y * obj.scale.z);

        _o.push_back(obj);                  /* Vector of objects */

        /* DEBUG - Print Message Params...  */
        std::cout << "   Object Position: " << obj.position.x << ", "
                                            << obj.position.y << ", "
                                            << obj.position.z << std::endl;

        std::cout << "   Object Scale: " << obj.scale.x << ", "
                                         << obj.scale.y << ", "
                                         << obj.scale.z << std::endl;

        /* Display obstacles as cube-type markers */
        visualization_msgs::Marker obstacle_marker;

        obstacle_marker.header.seq = 0;
        obstacle_marker.header.frame_id = "enu";
        obstacle_marker.header.stamp = ros::Time::now();

        obstacle_marker.id = o.id;
        obstacle_marker.type = visualization_msgs::Marker::CUBE;
        obstacle_marker.action = visualization_msgs::Marker::ADD;

        obstacle_marker.scale = o.scale;
        obstacle_marker.pose.position = o.position;

        obstacle_marker.color.a = 0.6;
        obstacle_marker.color.r = 1;
        obstacle_marker.color.g = 1;
        obstacle_marker.color.b = 1;

        markers.markers.push_back(obstacle_marker);
        if(obstacle_marker.id > max_id)
            max_id = obstacle_marker.id;
    }
    objs.objects = _o;

    std::cout << "MAX: " << max_id << std::endl;
    for(int i = 0; i < _o.size(); i++)
        pubPC.publish(_o[i].cloud);

    pubObjPC.publish(objs);
    pubMarkers.publish(markers);
    pubBuoys.publish(allBuoys);

    //Elapsed time
    ROS_INFO_STREAM("LIDAR | Elapsed time: " << (ros::Time::now()-timer).toSec());
    ROS_INFO("**********************************************************");
}


/* Update odometry information */
void cb_odom(const nav_msgs::OdometryConstPtr &odom) {
    boatPose_enu = odom->pose.pose;
    boatTwist_enu = odom->twist.twist;
}

int main(int argc, char* argv[])
{
    /* TODO: Convert to Nodelet */
    ros::init(argc, argv, "occupancy_grid");
    ros::Time::init();

    ros::NodeHandle nh;
    ros::Subscriber sub1 = nh.subscribe("/velodyne_points", 1, cb_velodyne);
    ros::Subscriber sub2 = nh.subscribe("/odom", 1, cb_odom);

    pubGrid = nh.advertise<nav_msgs::OccupancyGrid>("ogrid_batcave", 10);
    pubMarkers = nh.advertise<visualization_msgs::MarkerArray>("/unclassified/objects/markers", 10);
    pubBuoys = nh.advertise<navigator_msgs::BuoyArray>("/unclassified/objects", 10);
    pubObjPC = nh.advertise<navigator_msgs::Objects>("/occupancy_grid/objects", 10);
    pubPC = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points/oa_debug", 10);

    ros::spin();
    return 0;
}
