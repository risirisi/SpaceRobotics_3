#!/usr/bin/env python3

import math
import random
from enum import Enum 

import numpy as np # risi2 
from geometry_msgs.msg import PoseStamped # risi
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose # risi
import os # risi 
import tflite_runtime.interpreter as tflite # risi 
from rclpy.duration import Duration # risi 


import cv2  # OpenCV2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    # Add more!


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False

        # Marker for artifact locations
        # See https://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_artifacts_ = Marker()
        self.marker_artifacts_.header.frame_id = "map"
        self.marker_artifacts_.ns = "artifacts"
        self.marker_artifacts_.id = 0
        self.marker_artifacts_.type = Marker.SPHERE_LIST
        self.marker_artifacts_.action = Marker.ADD
        self.marker_artifacts_.pose.position.x = 0.0
        self.marker_artifacts_.pose.position.y = 0.0
        self.marker_artifacts_.pose.position.z = 0.0
        self.marker_artifacts_.pose.orientation.x = 0.0
        self.marker_artifacts_.pose.orientation.y = 0.0
        self.marker_artifacts_.pose.orientation.z = 0.0
        self.marker_artifacts_.pose.orientation.w = 1.0
        self.marker_artifacts_.scale.x = 1.5
        self.marker_artifacts_.scale.y = 1.5
        self.marker_artifacts_.scale.z = 1.5
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        self.marker_artifacts_.lifetime = Duration(seconds=0.0).to_msg() # risi 
        self._last_marker_pts = []  # cache last published points - risi 

        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Remember the artifact locations
        # Array of type geometry_msgs.Point
        self.artifact_locations_ = []

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ################# PERECEPTION - RISI #######################

        # Camera intrinsics 
        self.declare_parameter('fx', 554.256)
        self.declare_parameter('fy', 554.256)
        self.declare_parameter('cx', 320.5)
        self.declare_parameter('cy', 240.5)
        self.fx = float(self.get_parameter('fx').value)
        self.fy = float(self.get_parameter('fy').value)
        self.cx = float(self.get_parameter('cx').value)
        self.cy = float(self.get_parameter('cy').value)

        # Track latest headers/frames
        self.last_rgb_header_ = None
        self.last_depth_ = None
        self.last_cam_frame_ = 'camera_link'

        # Depth subscription 
        self.depth_sub_ = self.create_subscription(Image, 'camera/depth/image', self.depth_callback, 1)

        ######################### PERCEPTION T1 end - RISI ###############################

        ########################## T2- Risi ###########################

        # TFLite verifier
        self.declare_parameter('tflite_model_path', '')
        self.declare_parameter('tflite_labels_path', '')
        self.declare_parameter('tm_conf_thresh', 0.80)
        self.tm_conf_thresh = float(self.get_parameter('tm_conf_thresh').value)

        tflite_path = self.get_parameter('tflite_model_path').value
        labels_path = self.get_parameter('tflite_labels_path').value

        self.tm_interpreter = None
        self.tm_input_index = None
        self.tm_output_index = None
        self.tm_input_size = (224, 224)
        self.tm_labels = []
        try:
            if tflite_path:
                self.tm_interpreter = tflite.Interpreter(model_path=tflite_path, num_threads=2)
                self.tm_interpreter.allocate_tensors()
                inp = self.tm_interpreter.get_input_details()[0]
                out = self.tm_interpreter.get_output_details()[0]
                self.tm_input_index = inp['index']
                self.tm_output_index = out['index']
                h, w = inp['shape'][1], inp['shape'][2]
                self.tm_input_size = (w, h)
                if labels_path and os.path.exists(labels_path):
                    with open(labels_path, 'r') as f:
                        self.tm_labels = [line.strip() for line in f if line.strip()]
                self.get_logger().warn(f"TFLite loaded. input={self.tm_input_size}, labels={self.tm_labels}")
            else:
                self.get_logger().warn("No TFLite model path provided — verifier disabled.")
        except Exception as e:
            self.get_logger().warn(f"TFLite init failed: {e}")
            self.tm_interpreter = None

        # Artifact fusion params
        self.declare_parameter('artifact_merge_radius', 1.0)   # meters
        self.declare_parameter('artifact_ema_alpha', 0.3)      # 0..1, higher = more responsive
        self.declare_parameter('artifact_ttl_sec', 120.0)       
        self.declare_parameter('artifact_min_hits', 2)  

        self.declare_parameter('artifact_hist_len', 10) #T3 start risi 
        self.declare_parameter('artifact_spread_max', 0.6)  # meters
        self.hist_len = int(self.get_parameter('artifact_hist_len').value)
        self.spread_max = float(self.get_parameter('artifact_spread_max').value) #T3 end risi        

        self.merge_r = float(self.get_parameter('artifact_merge_radius').value)
        self.ema_a  = float(self.get_parameter('artifact_ema_alpha').value)
        self.ttl_s  = float(self.get_parameter('artifact_ttl_sec').value)
        self.min_hits = int(self.get_parameter('artifact_min_hits').value)

        self.artifacts = []  # list of dicts: {'x','y','z','hits','last'}

        # publish poses for malindu 
        from geometry_msgs.msg import PoseArray, Pose  #T3 Risi
        self.artifact_pose_array_pub_ = self.create_publisher(PoseArray, 'artifact_poses', 1)

        # republish poses at 2 Hz so planner can subscribe anytime
        self.pose_array_timer_ = self.create_timer(0.5, self.publish_artifact_pose_array)       #T3 risi 
                
        ########################## T2- Risi ###########################


        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        # self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL) # risi edit
        self.declare_parameter('print_feedback', False) # risi edit
        self.declare_parameter('enable_planning', False) #risi edit


        # Publisher for the goal pose visualisation
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing - RISI edit

        # self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        # self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)
        # self.computer_vision_model_ = cv2.CascadeClassifier(self.get_parameter('computer_vision_model_filename').value)
        # self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)

        # Prepare image processing - RISI edit 

        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)

        model_path = self.get_parameter('computer_vision_model_filename').value
        if not model_path or not os.path.exists(model_path):
            self.get_logger().warn("⚠️ No valid cascade model file found; detector will be disabled.")
            self.computer_vision_model_ = None
        else:
            self.computer_vision_model_ = cv2.CascadeClassifier(model_path)
            if self.computer_vision_model_.empty():
                self.get_logger().error(f"❌ Failed to load cascade from {model_path}; detector disabled.")
                self.computer_vision_model_ = None

        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)

        #####################################



        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)
    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

        # Lookup the latest transform
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        self.get_logger().warn(f'Pose: {pose}')

        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        """New map received, so update x and y limits"""

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x, 
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')
    
    def image_callback(self, image_msg):
        """
        Recieve an RGB image.
        Use this method to detect artifacts of interest.
        
        A simple method has been provided to begin with for detecting stop signs (which is not what we're actually looking for) 
        adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
        """
    
        # # Copy the image message to a cv image
        # # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        # image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # # Create a grayscale version (some simple models use this)
        # # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Retrieve the pre-trained model
        # stop_sign_model = self.computer_vision_model_

        # # Detect artifacts in the image
        # # The minSize is used to avoid very small detections that are probably noise
        # detections = stop_sign_model.detectMultiScale(image, minSize=(20,20))

        # # You can set "artifact_found_" to true to signal to "main_loop" that you have found a artifact
        # # You may want to communicate more information
        # # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
        # # with a mutex
        # # "artifact_found_" doesn't need a mutex because it's an atomic
        # num_detections = len(detections)

        # if num_detections > 0:
        #     self.artifact_found_ = True
        # else:
        #     self.artifact_found_ = False

        # # Draw a bounding box rectangle on the image for each detection
        # for(x, y, width, height) in detections:
        #     cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 0), 5)

        # # Publish the image with the detection bounding boxes
        # image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8")
        # self.image_detections_pub_.publish(image_detection_message)

        # if self.artifact_found_:
        #     self.get_logger().info('Artifact found!')
        #     self.localise_artifact()

        #################### PERCEPTION RISII ####################
        try:
           
            image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.last_rgb_header_ = image_msg.header
            if image_msg.header.frame_id:
                self.last_cam_frame_ = image_msg.header.frame_id

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 60, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

            # ignore masks
            H, W = image.shape[:2]
            ignore = np.zeros((H, W), np.uint8)
            cv2.rectangle(ignore, (0, 0), (W, int(0.12 * H)), 255, -1)
            cv2.rectangle(ignore, (0, int(0.70 * H)), (int(0.45 * W), H), 255, -1)
            edges[ignore > 0] = 0

            # proposals → raw_boxes
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_boxes = []
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area < 1000 or w < 20 or h < 20:
                    continue
                ar = w / max(1.0, h)
                if ar < 0.5 or ar > 2.5:
                    continue
                keep = True
                if self.last_depth_ is not None:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(W, x + w), min(H, y + h)
                    roi_d = self.last_depth_[y1:y2, x1:x2]
                    if roi_d.size > 0:
                        d = np.nanmedian(roi_d)
                        if not (np.isfinite(d) and 0.3 <= d <= 8.0):
                            keep = False
                        else:
                            phys_w = (w / self.fx) * float(d)
                            phys_h = (h / self.fy) * float(d)
                            if phys_w < 0.06 or phys_w > 1.2 or phys_h < 0.06 or phys_h > 1.2:
                                keep = False
                if keep:
                    raw_boxes.append((x, y, w, h))

            # verify first
            verified_boxes, confs = [], []
            if getattr(self, 'tm_interpreter', None) is not None:
                for (x, y, w, h) in raw_boxes:
                    crop = image[max(0, y):y + h, max(0, x):x + w]
                    conf = self.tm_predict_artifact_conf(crop)
                    if conf >= self.tm_conf_thresh:
                        verified_boxes.append((x, y, w, h))
                        confs.append(conf)
            else:
                verified_boxes = raw_boxes
                confs = [float(b[2]*b[3]) for b in raw_boxes]

            detections = self.nms_xywh(verified_boxes, confs, 0.5) \
                if verified_boxes and hasattr(self, 'nms_xywh') else verified_boxes

            # draw + choose best
            best = None
            for (x, y, w, h) in detections:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                area = w * h
                if best is None or area > best[4]:
                    best = (x, y, w, h, area)

            # always publish the debug image
            self.image_detections_pub_.publish(self.cv_bridge_.cv2_to_imgmsg(image, encoding='bgr8'))

            # localise if have a best box
            self.artifact_found_ = len(detections) > 0
            if self.artifact_found_ and best is not None:
                x, y, w, h, _ = best
                u = float(x + 0.5 * w)
                v = float(y + 0.5 * h)
                depth_m = None
                if self.last_depth_ is not None:
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(self.last_depth_.shape[1], int(x + w)), min(self.last_depth_.shape[0], int(y + h))
                    roi = self.last_depth_[y1:y2, x1:x2]
                    if roi.size > 0:
                        d = np.nanmedian(roi)
                        if np.isfinite(d) and 0.05 < d < 10.0:
                            depth_m = float(d)
                self.get_logger().info('Artifact found! Localising')
                if depth_m is not None:
                    self.localise_artifact(u=u, v=v, depth_m=depth_m, cam_frame=self.last_cam_frame_)
            else:
                self.artifact_found_ = False

        except Exception as e:
            # publish raw frame for debugging
            self.get_logger().error(f"image_callback error: {e}")
            try:
                raw = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                self.image_detections_pub_.publish(self.cv_bridge_.cv2_to_imgmsg(raw, encoding='bgr8'))
            except Exception:
                pass

        ###################### PERCEPTION RISII ##################


    # def localise_artifact(self): ###### old code
    #     """
    #     INCOMPLETE:
    #     Compute the location of the artifact
    #     Save it to a list, publish rviz marker
    #     This version just uses the robot location rather than the artifact location
    #     You can find other examples of using RViz markers in the previous assignments template code
    #     """

    #     # Current location of the robot
    #     robot_pose = self.get_pose_2d()

    #     if robot_pose == None:
    #         self.get_logger().warn(f'localise_artifact: robot_pose is None.')
    #         return

    #     # Compute the location of the artifact
    #     # This is currently INCOMPLETE
    #     point = Point()
    #     point.x = robot_pose.x
    #     point.y = robot_pose.y
    #     point.z = 1.0

    #     # Save it
    #     self.artifact_locations_.append(point)

    #     # Publish the markers
    #     self.publish_artifact_markers()

    ####### NEW code for def localise - Risi ######################

    def localise_artifact(self, u: float, v: float, depth_m: float, cam_frame: str):
        """
        Compute artifact 3D position in map frame from pixel (u,v) and depth
        Falls back to 2.0 m if depth missing. Saves and publishes markers
        """
        # Must have a valid metric depth
        if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0.0:
            return

        # Pinhole back-projection in camera frame
        Xc = (u - self.cx) / self.fx * depth_m
        Yc = (v - self.cy) / self.fy * depth_m
        Zc = depth_m

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = cam_frame if cam_frame else 'camera_link'
        ps.pose.position.x = float(Xc)
        ps.pose.position.y = float(Yc)
        ps.pose.position.z = float(Zc)
        ps.pose.orientation.w = 1.0

        try:
            t = self.tf_buffer.lookup_transform(
                'map',                      # target
                ps.header.frame_id,         # source (camera_link)
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            pose_map = do_transform_pose(ps.pose, t)   # Pose (not PoseStamped)
        except TransformException as ex:
            self.get_logger().warn(f'localise_artifact TF failed: {ex}')
            return

        # Merge into stable list and publish
        self._register_artifact(
            pose_map.position.x,
            pose_map.position.y,
            pose_map.position.z
        )
        self.publish_artifact_markers()

    ########################################################################


    def publish_artifact_markers(self):  ############## T3 risi ###########
        """Publish ONE sphere per stable artifact"""
        pts = []
        for a in self.artifacts:
            if a['hits'] < self.min_hits or len(a.get('hist', [])) < max(2, self.min_hits):
                continue
            # Spread check for stability
            xs, ys, _ = zip(*a['hist'])
            dx = np.array(xs) - a['x']
            dy = np.array(ys) - a['y']
            spread = float(np.sqrt(np.mean(dx*dx + dy*dy)))
            if spread > self.spread_max:
                continue

            p = Point(); p.x, p.y, p.z = a['x'], a['y'], a['z']
            pts.append(p)

        # If we have stable points, update cache and publish; else re-publish last ones
        if pts:
            self._last_marker_pts = pts
        elif self._last_marker_pts:
            pts = self._last_marker_pts
        else:
            # Nothing to publish yet
            return

        self.marker_artifacts_.points = pts
        marker_array = MarkerArray()
        marker_array.markers = [self.marker_artifacts_]
        self.marker_pub_.publish(marker_array)
        


    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        ####### risi edit ########
        #  safety for print_feedback param 
        print_feedback = self.get_parameter('print_feedback').get_parameter_value().bool_value
        if print_feedback:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        ########################


        # Decide whether to show feedback or not
        if self.get_parameter('print_feedback').value:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        # Send goal to action server
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        # Goal accepted: get result when it's completed
        self.get_logger().warn(f'Goal accepted')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""

        result = future.result().result
        self.get_logger().info(f'Goal reached!')
        self.ready_for_next_goal_ = True


    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location
        goal_valid = False
        while not goal_valid:
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
            else:
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

     ################# PERCEPTION - RISII
    def depth_callback(self, depth_msg: Image):
        """Store depth image in meters"""
        try:
            depth = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                self.last_depth_ = depth.astype(np.float32) * 0.001  # mm to m
            else:
                self.last_depth_ = depth.astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'Depth convert failed: {e}')

    ######################## PERCEPTION RISII

    ################################## T2 - Risii

    def tm_predict_artifact_conf(self, bgr_crop: np.ndarray) -> float:
        if self.tm_interpreter is None or bgr_crop.size == 0:
            return 0.0
        try:
            w, h = self.tm_input_size
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h)).astype(np.float32) / 255.0
            inp = np.expand_dims(resized, axis=0)
            self.tm_interpreter.set_tensor(self.tm_input_index, inp)
            self.tm_interpreter.invoke()
            out = self.tm_interpreter.get_tensor(self.tm_output_index)[0]

            # pick positive class index:
            idx_art = None
            if self.tm_labels:
                # look for 'artifact' or 'positive' in label text
                for i, lab in enumerate(self.tm_labels):
                    lab_l = lab.lower()
                    if 'artifact' in lab_l or 'positive' in lab_l or 'postive' in lab_l:
                        idx_art = i
                        break
            if idx_art is None:
                # assume class 0 is positive 
                idx_art = 0
            return float(out[idx_art])
        except Exception as e:
            self.get_logger().warn(f"TFLite inference failed: {e}")
            return 0.0

    def _register_artifact(self, x, y, z):
        """Merge detection; keep recent history; update robust center"""
        now = self.get_clock().now().nanoseconds / 1e9

        # Find nearest existing artifact by distance
        best_i, best_d = None, 1e9
        for i, a in enumerate(self.artifacts):
            d = ((a['x'] - x)**2 + (a['y'] - y)**2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i

        # Merge if within proximity using merge radius
        if best_d <= self.merge_r and best_i is not None:
            a = self.artifacts[best_i]
            # Append to history
            a['hist'].append((x, y, z))
            if len(a['hist']) > self.hist_len:
                a['hist'].pop(0)

            # (median)
            xs, ys, zs = zip(*a['hist'])
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            med_z = float(np.median(zs))

            # Use exponential moving average (EMA) to converge to this position
            a['x'] = (1 - self.ema_a) * a['x'] + self.ema_a * med_x
            a['y'] = (1 - self.ema_a) * a['y'] + self.ema_a * med_y
            a['z'] = (1 - self.ema_a) * a['z'] + self.ema_a * med_z

            # Increase hit count and update the timestamp
            a['hits'] += 1
            a['last'] = now
        else:
            # New artifact detection
            self.artifacts.append({
                'x': x, 'y': y, 'z': z,
                'hits': 1, 'last': now,
                'hist': [(x, y, z)]
            })

        # Remove stale artifacts
        self.artifacts = [a for a in self.artifacts if (now - a['last']) <= self.ttl_s]




    ########################################### T2 - Risii

    ############ T3 - RIsi #################

    def publish_artifact_pose_array(self):
        """Continuously publish artifact poses for planner."""
        from geometry_msgs.msg import PoseArray, Pose  
        pa = PoseArray()
        pa.header.frame_id = 'map'
        pa.header.stamp = self.get_clock().now().to_msg()
        for a in self.artifacts:
            if a['hits'] >= self.min_hits:
                pose = Pose()
                pose.position.x = a['x']
                pose.position.y = a['y']
                pose.position.z = a['z']
                pose.orientation.w = 1.0  # neutral yaw
                pa.poses.append(pose)
        self.artifact_pose_array_pub_.publish(pa)
        
    ################# T3 Risi ###############################


    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        See https://docs.nav2.org/concepts/index.html
        """
        
        if not self.get_parameter('enable_planning').get_parameter_value().bool_value:
            return  # planning disabled; do nothing


        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        #######################################################
        # Update flags related to the progress of the current planner

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.get_logger().info('Successfully reached first artifact!')
            self.reached_first_artifact_ = True
        if self.planner_type_ == PlannerType.RETURN_HOME:
            self.get_logger().info('Successfully returned home!')
            self.returned_home_ = True

        #######################################################
        # Select the next planner to execute
        # Update this logic as you see fit!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.RANDOM_GOAL

        #######################################################
        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()


        #######################################################

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)