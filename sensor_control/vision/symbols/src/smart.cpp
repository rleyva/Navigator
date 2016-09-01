#include "std_msgs/String.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include "FrameProc.h"
#include "ShapeFind.h"

#include <vector>

#include "DebugWindow.h"
#include "opencv2/opencv.hpp"

#include "navigator_msgs/GetDockShape.h"
#include "navigator_msgs/DockShapes.h"
#include "std_srvs/SetBool.h"




//Uncomment to view a slider bar for faking the camera center
//#define USE_FAKE_SYMBOL 

#ifdef USE_FAKE_SYMBOL
#include <cv.h>
#include <highgui.h>
#endif

class ImageSearcher {
 private:
  ros::NodeHandle n;
  ros::Subscriber sub;
  ros::ServiceServer service;
  navigator_msgs::DockShapes syms;                            // latest frame
  std::vector<navigator_msgs::DockShapes> frameSymbolHolder;  // Previous frames
  std::vector<navigator_msgs::DockShape> possibleSymbols;
  std::string possibleShapes[3] = {navigator_msgs::DockShape::CROSS, navigator_msgs::DockShape::TRIANGLE,
                                  navigator_msgs::DockShape::CIRCLE};
  std::string possibleColors[3] = {navigator_msgs::DockShape::RED, navigator_msgs::DockShape::BLUE, navigator_msgs::DockShape::GREEN};
  ros::ServiceServer serviceCommand;
  int counter[3 * 3];
  int frames;
  bool active;
  int lastCallFrame;

  //For debug
  #ifdef USE_FAKE_SYMBOL
  navigator_msgs::DockShape fakeSymbol;
  #endif
 public:
  ImageSearcher() {
    frames = 0;
    lastCallFrame = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        navigator_msgs::DockShape holdSym;
        holdSym.Shape = possibleShapes[i];
        holdSym.Color = possibleColors[j];
        holdSym.CenterX = 0;
        holdSym.CenterY = 0;
        possibleSymbols.push_back(holdSym);
        counter[i * j] = 0;
      }
    }
    syms = navigator_msgs::DockShapes();
    sub = n.subscribe("/dock_shapes/found_shapes", 1000, &ImageSearcher::chatterCallback, this);
    serviceCommand = n.advertiseService("/dock_shapes/runsmart", &ImageSearcher::getShapeController, this);
    service = n.advertiseService("/dock_shapes/GetShape", &ImageSearcher::getShape, this);
    active = false;
    
    #ifdef USE_FAKE_SYMBOL
    //For fake testing
    fakeSymbol.CenterX = 100;
    fakeSymbol.CenterY = 300;
    fakeSymbol.Color = "BLUE";
    fakeSymbol.Shape = "CROSS";
    fakeSymbol.img_width = 1290;
    cv::namedWindow("Testing", 1);  
    cv::createTrackbar("SetCenterX", "Testing", (int *) &fakeSymbol.CenterX,fakeSymbol.img_width);
    #endif
  }

  float mean(int val, int size) { return val / size; }
  void shapeChecker(const navigator_msgs::DockShapes &symbols) {
    if (!active) return;
    for (int i = 0; i < symbols.list.size(); i++) {
      for (int k = 0; k < possibleSymbols.size(); k++) {
        if (symbols.list[i].Shape == possibleSymbols[k].Shape && symbols.list[i].Color == possibleSymbols[k].Color) {
          if (frames < 10) {
            possibleSymbols[k].CenterX += symbols.list[i].CenterX;
            possibleSymbols[k].CenterY += symbols.list[i].CenterY;
            counter[k]++;
          } else if (counter[k] > 0 && std::abs(symbols.list[i].CenterX - mean(possibleSymbols[k].CenterX, counter[k])) < 100 &&
                     std::abs(symbols.list[i].CenterY - mean(possibleSymbols[k].CenterY, counter[k])) < 100) {
            possibleSymbols[k].CenterX += symbols.list[i].CenterX;
            possibleSymbols[k].CenterY += symbols.list[i].CenterY;
            counter[k]++;
          }
        }
      }
    }
  }

  void chatterCallback(const navigator_msgs::DockShapes &symbols) {
    if (!active) return;
    frames++;
    syms = symbols;
    frameSymbolHolder.push_back(symbols);
    shapeChecker(symbols);
  }

  bool getShape(navigator_msgs::GetDockShape::Request &req, navigator_msgs::GetDockShape::Response &res) {

    #ifdef USE_FAKE_SYMBOL
    std::cout << "GetShape" << std::endl;
    //Fake response for testing 
    res.symbol = fakeSymbol;
    std::cout << "Center = " << fakeSymbol.CenterX << std::endl;
    return true;
    #endif

    if (!active) {
        res.success=true;
        return false;
    }
    if (frames < 10) {
      std::cout << "Too small of sample frames" << frames << std::endl;
      res.success=true;
      return false;
    }
    for (int j = 0; j < syms.list.size(); j++) {
      for (int i = 0; i < possibleSymbols.size(); i++) {
        if (req.Shape == possibleSymbols[i].Shape && req.Color == possibleSymbols[i].Color && syms.list[j].Shape == req.Shape &&
            syms.list[j].Color == req.Color && counter[i] > 10) {
          if (std::abs(syms.list[j].CenterX - mean(possibleSymbols[i].CenterX, counter[i])) < 100 &&
              std::abs(syms.list[j].CenterY - mean(possibleSymbols[i].CenterY, counter[i])) < 100) {
            res.symbol = syms.list[j];
            lastCallFrame = frames;
            res.success=true;
            return true;
          }
        }
      }
    }
    res.success=true;
    return false;
  }

  bool getShapeController(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res) {
    std_srvs::SetBool msg;
    msg.request.data = req.data;
    active = req.data;

    ros::service::call("/dock_shapes/runvision", msg);
    std::cout << "Setting active to " << active << std::endl;
    res.success = true;
    return true;
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "dock_shape_filter");
  ImageSearcher imageSearcher;
  while (waitKey(50) == -1 && ros::ok()) {
    ros::spinOnce();
  }
  ros::spin();
  return 0;
}
