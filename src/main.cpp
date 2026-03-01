/** 
 * ─────────────────────────────────────────────────────────────────────────────
 * MIT License

 * Copyright (c) 2026 WM Nipun Dhananjaya

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * ───────────────────────────────────────────────────────────────────────────── 
*/


/** ────────────────────────────────────────────────────────────────────────────
 * @file: main.cpp
 * @brief: Main entry point for the object detection node.
 * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @company: Sintez.LLC
 * @date: 2026-03-01
 * 
 * This file contains the main() function that initializes the ROS2 node and 
 * starts the executor.
 * 
 * It creates an instance of the ObjectDetectionNode and adds it to a 
 * MultiThreadedExecutor, which allows the node to process callbacks 
 * concurrently if needed.  
 * 
 * The main function is straightforward and primarily responsible for 
 * setting up the ROS2 environment and starting the node.  
 * 
 * The actual logic of subscribing to the image topic, running inference, and 
 * publishing detections is implemented in the ObjectDetectionNode class, 
 * which is defined in object_detection_node.hpp and implemented in 
 * object_detection_node.cpp.  
 * ─────────────────────────────────────────────────────────────────────────────
*/

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_object_detection/object_detection_node.hpp"



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::executors::MultiThreadedExecutor executor;

  auto node = std::make_shared<nvidia::isaac_ros::object_detection::ObjectDetectionNode>(
    rclcpp::NodeOptions());

  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}