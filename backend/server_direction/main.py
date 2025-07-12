import cv2
import numpy as np
from ultralytics import YOLO
import math
import torch
import mediapipe as mp
from scipy.spatial.distance import euclidean
from collections import deque
import time
import warnings

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

class TargetTracker:
    """
    Robust target object tracking with position prediction and temporal consistency
    """
    def __init__(self, max_distance=150, min_confidence=0.5):
        self.tracked_targets = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.max_frames_lost = 8
        self.locked_target_id = None
        self.lock_threshold = 2
        self.lock_counter = 0
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_center_distance(self, box1, box2):
        """Calculate distance between centers of two bounding boxes"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return euclidean(center1, center2)
    
    def calculate_prediction_confidence(self, frames_lost):
        """Calculate prediction confidence based on frames lost"""
        return max(0.3, 0.8 - (frames_lost * 0.1))
    
    def predict_position(self, track):
        """Predict target position based on velocity and time elapsed"""
        if len(track['center_history']) < 2:
            return None
            
        # Calculate velocity from recent positions
        recent_positions = list(track['center_history'])[-3:]  # Use last 3 positions
        if len(recent_positions) < 2:
            return None
            
        # Simple velocity calculation
        velocity_x = (recent_positions[-1][0] - recent_positions[0][0]) / len(recent_positions)
        velocity_y = (recent_positions[-1][1] - recent_positions[0][1]) / len(recent_positions)
        
        # Predict position based on frames lost
        frames_lost = track['frames_lost']
        predicted_center = (
            recent_positions[-1][0] + velocity_x * frames_lost,
            recent_positions[-1][1] + velocity_y * frames_lost
        )
        
        # Create predicted bbox (maintain size from last detection)
        last_bbox = track['bbox']
        width = last_bbox[2] - last_bbox[0]
        height = last_bbox[3] - last_bbox[1]
        
        predicted_bbox = np.array([
            predicted_center[0] - width/2,
            predicted_center[1] - height/2,
            predicted_center[0] + width/2,
            predicted_center[1] + height/2
        ])
        
        return predicted_bbox
    
    def update(self, detections):
        """
        Update tracker with new detections and maintain tracking through brief losses
        """
        current_frame_time = time.time()
        
        # Mark all tracked targets as not seen this frame
        for target_id in self.tracked_targets:
            self.tracked_targets[target_id]['seen_this_frame'] = False
        
        # Match detections to existing tracks
        matched_detections = set()
        
        for detection_idx, (bbox, confidence, class_name) in enumerate(detections):
            if confidence < self.min_confidence:
                continue
                
            best_match_id = None
            best_match_score = float('inf')
            
            # Find best matching existing track
            for target_id, track in self.tracked_targets.items():
                if track['seen_this_frame']:
                    continue
                    
                # Use current bbox or predicted position for comparison
                compare_bbox = track['bbox']
                if track['frames_lost'] > 0:
                    predicted_bbox = self.predict_position(track)
                    if predicted_bbox is not None:
                        compare_bbox = predicted_bbox
                
                # Calculate matching score
                iou = self.calculate_iou(bbox, compare_bbox)
                center_dist = self.calculate_center_distance(bbox, compare_bbox)
                
                # Weighted score (lower is better)
                score = center_dist - (iou * 100)  # Favor high IoU more for targets
                
                if score < best_match_score and center_dist < self.max_distance:
                    best_match_score = score
                    best_match_id = target_id
            
            # Update existing track or create new one
            if best_match_id is not None:
                # Update existing track
                self.tracked_targets[best_match_id].update({
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': current_frame_time,
                    'seen_this_frame': True,
                    'frames_lost': 0,
                    'class_name': class_name
                })
                
                # Update center history
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.tracked_targets[best_match_id]['center_history'].append(center)
                
                matched_detections.add(detection_idx)
            else:
                # Create new track
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.tracked_targets[self.next_id] = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': current_frame_time,
                    'seen_this_frame': True,
                    'frames_lost': 0,
                    'center_history': deque([center], maxlen=10),
                    'first_seen': current_frame_time,
                    'class_name': class_name
                }
                matched_detections.add(detection_idx)
                self.next_id += 1
        
        # Update frames_lost for unmatched tracks
        to_remove = []
        for target_id, track in self.tracked_targets.items():
            if not track['seen_this_frame']:
                track['frames_lost'] += 1
                if track['frames_lost'] > self.max_frames_lost:
                    to_remove.append(target_id)
                    if self.locked_target_id == target_id:
                        print(f"🎯 Lost locked target {target_id}")
        
        # Remove lost tracks
        for target_id in to_remove:
            del self.tracked_targets[target_id]
            if self.locked_target_id == target_id:
                self.locked_target_id = None
                self.lock_counter = 0
    
    def get_best_target(self):
        """
        Get the best target to track (highest confidence, most stable)
        Returns tracking data even for temporarily lost targets
        """
        if not self.tracked_targets:
            return None, None
            
        # If we have a locked target, prioritize it
        if (self.locked_target_id is not None and 
            self.locked_target_id in self.tracked_targets):
            
            target_data = self.tracked_targets[self.locked_target_id]
            
            # For temporarily lost targets, use predicted position
            if target_data['frames_lost'] > 0:
                predicted_bbox = self.predict_position(target_data)
                if predicted_bbox is not None:
                    # Create a copy with predicted position
                    predicted_data = target_data.copy()
                    predicted_data['bbox'] = predicted_bbox
                    predicted_data['is_predicted'] = True
                    predicted_data['prediction_confidence'] = self.calculate_prediction_confidence(target_data['frames_lost'])
                    return self.locked_target_id, predicted_data
            
            return self.locked_target_id, target_data
        
        # Find best target based on multiple criteria
        best_target_id = None
        best_score = -1
        
        for target_id, track in self.tracked_targets.items():
            # Don't exclude temporarily lost targets
            confidence_score = track['confidence']
            stability_score = min(len(track['center_history']) / 5.0, 1.0)
            duration_score = min((time.time() - track['first_seen']) / 3.0, 1.0)
            
            # Penalty for lost frames but don't disqualify
            lost_penalty = max(0.2, 1.0 - (track['frames_lost'] * 0.15))
            
            # Weighted combined score
            combined_score = (confidence_score * 0.4 + 
                            stability_score * 0.3 + 
                            duration_score * 0.2 +
                            lost_penalty * 0.1)
            
            if combined_score > best_score:
                best_score = combined_score
                best_target_id = target_id
        
        # Lock the target if consistently the best
        if best_target_id is not None:
            if best_target_id == getattr(self, '_last_best_id', None):
                self.lock_counter += 1
                if self.lock_counter >= self.lock_threshold:
                    self.locked_target_id = best_target_id
                    print(f"🔒 Target {best_target_id} locked for tracking")
            else:
                self.lock_counter = 1
                self._last_best_id = best_target_id
        
        if best_target_id is not None:
            target_data = self.tracked_targets[best_target_id]
            
            # For temporarily lost targets, use predicted position
            if target_data['frames_lost'] > 0:
                predicted_bbox = self.predict_position(target_data)
                if predicted_bbox is not None:
                    predicted_data = target_data.copy()
                    predicted_data['bbox'] = predicted_bbox
                    predicted_data['is_predicted'] = True
                    predicted_data['prediction_confidence'] = self.calculate_prediction_confidence(target_data['frames_lost'])
                    return best_target_id, predicted_data
            
            return best_target_id, target_data
        
        return None, None
    
    def reset_lock(self):
        """Reset target lock"""
        self.locked_target_id = None
        self.lock_counter = 0
        print("🔓 Target tracking lock reset")

class PersonTracker:
    """
    Robust person tracking with ID assignment and consistency
    """
    def __init__(self, max_distance=100, min_confidence=0.6, target_tracker=None):
        self.tracked_persons = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.max_frames_lost = 5
        self.locked_person_id = None
        self.lock_threshold = 3
        self.lock_counter = 0
        self.target_tracker = target_tracker
        
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (bbox, confidence) tuples
        Focus on locked person if available
        """
        current_frame_time = time.time()
        
        # If we have a locked person, focus only on tracking them
        if self.locked_person_id is not None and self.locked_person_id in self.tracked_persons:
            locked_person = self.tracked_persons[self.locked_person_id]
            locked_person['seen_this_frame'] = False
            
            # Try to match the locked person with new detections
            best_match_detection = None
            best_match_score = float('inf')
            
            for detection_idx, (bbox, confidence) in enumerate(detections):
                if confidence < self.min_confidence:
                    continue
                    
                # Calculate matching score with locked person
                iou = self.target_tracker.calculate_iou(bbox, locked_person['bbox']) if self.target_tracker else 0
                center_dist = self.target_tracker.calculate_center_distance(bbox, locked_person['bbox']) if self.target_tracker else float('inf')
                
                # Weighted score (lower is better)
                score = center_dist - (iou * 50)  # Favor high IoU
                
                if score < best_match_score and center_dist < self.max_distance:
                    best_match_score = score
                    best_match_detection = (bbox, confidence)
            
            # Update locked person if found
            if best_match_detection is not None:
                bbox, confidence = best_match_detection
                locked_person.update({
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': current_frame_time,
                    'seen_this_frame': True,
                    'frames_lost': 0
                })
                
                # Update center history
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                locked_person['center_history'].append(center)
            else:
                # Locked person not found in this frame
                locked_person['frames_lost'] += 1
                if locked_person['frames_lost'] > self.max_frames_lost:
                    # Lost the locked person, reset everything
                    print(f"🔓 Lost locked person {self.locked_person_id}, resetting...")
                    self.tracked_persons.clear()
                    self.locked_person_id = None
                    self.lock_counter = 0
                    self.next_id = 0
            
            return  # Don't process other detections if we're focused on locked person
        
        # Standard tracking logic when no person is locked
        # Mark all tracked persons as not seen this frame
        for person_id in self.tracked_persons:
            self.tracked_persons[person_id]['seen_this_frame'] = False
        
        # Match detections to existing tracks
        matched_detections = set()
        
        for detection_idx, (bbox, confidence) in enumerate(detections):
            if confidence < self.min_confidence:
                continue
                
            best_match_id = None
            best_match_score = float('inf')
            
            # Find best matching existing track
            for person_id, track in self.tracked_persons.items():
                if track['seen_this_frame']:
                    continue
                    
                # Calculate matching score (combination of IoU and center distance)
                iou = self.target_tracker.calculate_iou(bbox, track['bbox']) if self.target_tracker else 0
                center_dist = self.target_tracker.calculate_center_distance(bbox, track['bbox']) if self.target_tracker else float('inf')
                
                # Weighted score (lower is better)
                score = center_dist - (iou * 50)  # Favor high IoU
                
                if score < best_match_score and center_dist < self.max_distance:
                    best_match_score = score
                    best_match_id = person_id
            
            # Update existing track or create new one
            if best_match_id is not None:
                # Update existing track
                self.tracked_persons[best_match_id].update({
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': current_frame_time,
                    'seen_this_frame': True,
                    'frames_lost': 0
                })
                
                # Update center history
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                if 'center_history' not in self.tracked_persons[best_match_id]:
                    self.tracked_persons[best_match_id]['center_history'] = deque(maxlen=10)
                self.tracked_persons[best_match_id]['center_history'].append(center)
                
                matched_detections.add(detection_idx)
            else:
                # Create new track
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.tracked_persons[self.next_id] = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': current_frame_time,
                    'seen_this_frame': True,
                    'frames_lost': 0,
                    'center_history': deque([center], maxlen=10),
                    'first_seen': current_frame_time
                }
                matched_detections.add(detection_idx)
                self.next_id += 1
        
        # Update frames_lost for unmatched tracks
        to_remove = []
        for person_id, track in self.tracked_persons.items():
            if not track['seen_this_frame']:
                track['frames_lost'] += 1
                if track['frames_lost'] > self.max_frames_lost:
                    to_remove.append(person_id)
        
        # Remove lost tracks
        for person_id in to_remove:
            del self.tracked_persons[person_id]
            if self.locked_person_id == person_id:
                self.locked_person_id = None
                self.lock_counter = 0
    
    def get_best_person(self):
        """
        Get the best person to track (highest confidence, most stable)
        """
        if not self.tracked_persons:
            return None, None
            
        # If we have a locked person and they're still being tracked, use them
        if (self.locked_person_id is not None and 
            self.locked_person_id in self.tracked_persons and
            self.tracked_persons[self.locked_person_id]['frames_lost'] == 0):
            
            person_data = self.tracked_persons[self.locked_person_id]
            return self.locked_person_id, person_data
        
        # Find best person based on multiple criteria
        best_person_id = None
        best_score = -1
        
        for person_id, track in self.tracked_persons.items():
            if track['frames_lost'] > 0:
                continue
                
            # Scoring based on confidence, stability, and tracking duration
            confidence_score = track['confidence']
            stability_score = min(len(track['center_history']) / 10.0, 1.0)
            duration_score = min((time.time() - track['first_seen']) / 5.0, 1.0)
            
            # Weighted combined score
            combined_score = (confidence_score * 0.4 + 
                            stability_score * 0.3 + 
                            duration_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_person_id = person_id
        
        # Lock the person if they've been consistently the best for several frames
        if best_person_id is not None:
            if best_person_id == getattr(self, '_last_best_id', None):
                self.lock_counter += 1
                if self.lock_counter >= self.lock_threshold:
                    self.locked_person_id = best_person_id
                    print(f"🔒 Person {best_person_id} locked for tracking")
            else:
                self.lock_counter = 1
                self._last_best_id = best_person_id
        
        if best_person_id is not None:
            return best_person_id, self.tracked_persons[best_person_id]
        
        return None, None
    
    def reset_lock(self):
        """Reset person lock"""
        self.locked_person_id = None
        self.lock_counter = 0
        print("🔓 Person tracking lock reset")

class RobustObjectNavigationSystem:
    def __init__(self, model_path='yolo11l.pt', target_item='bottle'):
        """
        Robust navigation system with accurate 3D spatial reasoning
        """
        # Check if CUDA is available and set device accordingly
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        if self.device != 'cpu':
            self.model.to(self.device)
            
        self.target_item = target_item
        
        # Initialize target tracker for reliable object tracking - lowered confidence
        self.target_tracker = TargetTracker(max_distance=120, min_confidence=0.25)
        
        # Initialize person tracker - lowered confidence
        self.person_tracker = PersonTracker(max_distance=80, min_confidence=0.4, target_tracker=self.target_tracker)
        
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_objectron = mp.solutions.objectron
        
        # Enhanced pose estimation
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher complexity for better accuracy
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Hand tracking for pointing gestures
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Multiple depth estimation methods
        self.depth_methods = ['bbox_ratio', 'pose_geometry', 'relative_size', 'stereo_cues']
        
        # Calibrated camera parameters (you should calibrate your specific camera)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
        
        # More accurate object dimensions with variance
        self.object_dimensions = {
            'person': {'height': (160, 180), 'width': (35, 55), 'depth': (20, 30)},
            'bottle': {'height': (7, 10), 'width': (5, 8), 'depth': (5, 8)},
            'cup': {'height': (8, 12), 'width': (6, 10), 'depth': (6, 10)},
            'cell phone': {'height': (12, 18), 'width': (6, 8), 'depth': (0.5, 2)},
            'book': {'height': (18, 25), 'width': (12, 20), 'depth': (1, 4)},
            'laptop': {'height': (20, 30), 'width': (25, 40), 'depth': (1, 3)},
            'chair': {'height': (70, 90), 'width': (40, 60), 'depth': (40, 60)},
            'couch': {'height': (60, 80), 'width': (150, 200), 'depth': (70, 90)}
        }
        
        # Tracking and filtering - separate filters for different object types
        self.position_filter = deque(maxlen=15)
        self.orientation_filter = deque(maxlen=10)
        
        # Object-specific depth filters to prevent cross-contamination
        self.depth_filters = {
            'person': deque(maxlen=8),
            'bottle': deque(maxlen=8),
            'cup': deque(maxlen=8),
            'cell phone': deque(maxlen=8),
            'book': deque(maxlen=8),
            'laptop': deque(maxlen=8),
            'chair': deque(maxlen=8),
            'couch': deque(maxlen=8),
            'default': deque(maxlen=8)  # fallback for unknown objects
        }
        
        # Navigation state
        self.last_instruction_time = 0
        self.instruction_cooldown = 0.8
        self.navigation_history = deque(maxlen=20)
        
        # Confidence thresholds - lowered for better object detection
        self.min_detection_confidence = 0.2
        self.min_tracking_confidence = 0.5
        
        # Coordinate system: OpenCV camera coordinates
        # X: right, Y: down, Z: forward (into the scene)
        
    def _get_device(self):
        """Determine best available device"""
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def extract_person_roi(self, frame, bbox, padding=20):
        """
        Extract region of interest for a person with padding
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        roi = frame[y1:y2, x1:x2]
        offset = (x1, y1)
        
        return roi, offset
    
    def adjust_landmarks_to_full_frame(self, landmarks, offset, roi_shape, full_frame_shape):
        """
        Adjust MediaPipe landmarks from ROI coordinates to full frame coordinates
        """
        if not landmarks:
            return None
            
        offset_x, offset_y = offset
        roi_h, roi_w = roi_shape[:2]
        
        # Create a new landmark list with adjusted coordinates
        # We'll modify the landmarks in place since MediaPipe landmarks are mutable
        for landmark in landmarks.landmark:
            # Convert from ROI normalized coordinates to full frame normalized coordinates
            absolute_x = landmark.x * roi_w + offset_x
            absolute_y = landmark.y * roi_h + offset_y
            
            # Normalize to full frame
            landmark.x = absolute_x / full_frame_shape[1]
            landmark.y = absolute_y / full_frame_shape[0]
        
        return landmarks
    
    def estimate_depth_multi_method(self, bbox, object_class, pose_landmarks=None, frame_shape=None):
        """
        Multi-method depth estimation with confidence weighting
        """
        depths = []
        confidences = []
        
        # Method 1: Bounding box ratio with multiple dimensions
        if object_class in self.object_dimensions:
            x1, y1, x2, y2 = bbox
            pixel_height = y2 - y1
            pixel_width = x2 - x1
            
            # Use height estimation
            dim_range = self.object_dimensions[object_class]['height']
            avg_height = (dim_range[0] + dim_range[1]) / 2
            focal_length = self.camera_matrix[1, 1]
            
            depth_height = (avg_height * focal_length) / pixel_height
            depths.append(depth_height)
            confidences.append(0.7)
            
            # Use width estimation for cross-validation
            dim_range_w = self.object_dimensions[object_class]['width']
            avg_width = (dim_range_w[0] + dim_range_w[1]) / 2
            focal_length_x = self.camera_matrix[0, 0]
            
            depth_width = (avg_width * focal_length_x) / pixel_width
            depths.append(depth_width)
            confidences.append(0.6)
        
        # Method 2: Pose-based depth (for persons)
        if object_class == 'person' and pose_landmarks and frame_shape:
            pose_depth = self.estimate_depth_from_pose(pose_landmarks, frame_shape)
            if pose_depth:
                depths.append(pose_depth)
                confidences.append(0.8)
        
        # Method 3: Relative size comparison
        if len(depths) > 0:
            # Use the average of existing methods
            relative_depth = np.mean(depths)
            depths.append(relative_depth)
            confidences.append(0.5)
        
        # Weighted average
        if depths:
            weights = np.array(confidences) / np.sum(confidences)
            final_depth = np.average(depths, weights=weights)
            
            # Apply object-specific filtering to prevent cross-contamination
            filter_key = object_class if object_class in self.depth_filters else 'default'
            depth_filter = self.depth_filters[filter_key]
            
            depth_filter.append(final_depth)
            if len(depth_filter) > 3:
                filtered_depth = np.median(list(depth_filter))
                # Debug: Show which filter is being used
                print(f"Depth estimation for {object_class}: {final_depth:.1f}cm -> {filtered_depth:.1f}cm (filter: {filter_key}, samples: {len(depth_filter)})")
                return filtered_depth
            else:
                print(f"Depth estimation for {object_class}: {final_depth:.1f}cm (filter: {filter_key}, samples: {len(depth_filter)})")
                return final_depth
        
        return None
    
    def estimate_depth_from_pose(self, pose_landmarks, frame_shape):
        """
        Estimate depth using pose geometry
        """
        if not pose_landmarks:
            return None
            
        # Get key landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        # Calculate shoulder width in pixels
        height, width = frame_shape[:2]
        shoulder_width_px = abs(right_shoulder.x - left_shoulder.x) * width
        
        # Average human shoulder width: 40cm
        avg_shoulder_width = 40  # cm
        focal_length = self.camera_matrix[0, 0]
        
        if shoulder_width_px > 0:
            depth = (avg_shoulder_width * focal_length) / shoulder_width_px
            return depth
        
        return None
    
    def get_accurate_3d_position(self, bbox, depth, frame_shape):
        """
        Convert 2D detection to accurate 3D coordinates
        """
        if depth is None:
            return None
            
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Camera intrinsics
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        
        # Convert to 3D camera coordinates
        # X: right, Y: down, Z: forward
        x_cam = (center_x - cx) * depth / fx
        y_cam = (center_y - cy) * depth / fy
        z_cam = depth
        
        return np.array([x_cam, y_cam, z_cam])
    
    def get_robust_user_orientation(self, pose_landmarks, frame_shape, target_3d=None, person_3d=None):
        """
        Get accurate user orientation using facial features as primary method
        Falls back to body pose if face is not visible
        Enhanced logic to consider target position when determining facing direction
        """
        if not pose_landmarks:
            return None
            
        # Get key landmarks
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]
        left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # Check face visibility for primary method
        face_visible = (left_eye.visibility > 0.6 and right_eye.visibility > 0.6 and 
                       nose.visibility > 0.7)
        
        # Method 1: FACIAL FEATURE TRACKING (Primary - most accurate)
        if face_visible:
            # Calculate face direction using eye-nose geometry
            eye_center = np.array([(left_eye.x + right_eye.x) / 2, 
                                 (left_eye.y + right_eye.y) / 2])
            
            # Face direction vector (from eye center to nose)
            face_direction = np.array([nose.x - eye_center[0], 
                                     nose.y - eye_center[1]])
            
            # Calculate facing angle from face direction
            face_angle = math.atan2(face_direction[1], face_direction[0])
            
            # Determine if person is facing towards or away from camera
            # If eyes are clearly visible, person is likely facing camera
            eye_visibility_avg = (left_eye.visibility + right_eye.visibility) / 2
            
            # Check ear visibility to determine profile orientation
            left_ear_visible = left_ear.visibility > 0.5
            right_ear_visible = right_ear.visibility > 0.5
            
            if eye_visibility_avg > 0.8 and not (left_ear_visible and right_ear_visible):
                # Facing camera - use eye line for orientation
                eye_line = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
                eye_angle = math.atan2(eye_line[1], eye_line[0])
                # Facing direction is perpendicular to eye line
                facing_angle = eye_angle + math.pi/2
                facing_camera = True
                confidence = 0.9
            elif left_ear_visible and not right_ear_visible:
                # Left profile - facing right
                facing_angle = 0  # Facing right (positive X direction)
                facing_camera = False
                confidence = 0.8
            elif right_ear_visible and not left_ear_visible:
                # Right profile - facing left  
                facing_angle = math.pi  # Facing left (negative X direction)
                facing_camera = False
                confidence = 0.8
            else:
                # Use nose-eye vector for general direction
                facing_angle = face_angle + math.pi/2  # Perpendicular to face direction
                facing_camera = eye_visibility_avg > 0.7
                confidence = 0.7
                
        # Method 2: ENHANCED SHOULDER FALLBACK (when face not clearly visible)
        else:
            torso_center = np.array([(left_shoulder.x + right_shoulder.x) / 2,
                                   (left_shoulder.y + right_shoulder.y) / 2])
            
            # Calculate shoulder vector (perpendicular to facing direction)
            shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, 
                                     right_shoulder.y - left_shoulder.y])
            
            # Use nose position relative to torso for orientation
            nose_offset_x = nose.x - torso_center[0]
            
            shoulder_angle = math.atan2(shoulder_vector[1], shoulder_vector[0])
            
            # IMPROVED LOGIC: Consider target position when face is not visible
            if target_3d is not None and person_3d is not None:
                # Calculate target direction in camera coordinates
                target_vector = target_3d - person_3d
                target_angle_camera = math.atan2(target_vector[0], target_vector[2])  # X over Z
                
                # Check if target is in front of the camera (positive Z)
                target_in_front_of_camera = target_vector[2] > 0
                
                # Determine facing based on nose offset and target position
                if abs(nose_offset_x) < 0.03:  # Nose more centered - likely facing camera
                    facing_angle = shoulder_angle + math.pi/2
                    facing_camera = True
                    # Additional check: if target is behind camera but user seems to face camera,
                    # it means user is facing away from target
                    if not target_in_front_of_camera:
                        facing_camera = True  # Still facing camera, but target is behind
                else:
                    # Nose offset suggests profile or back view
                    # Use shoulder orientation to determine facing direction
                    facing_angle = shoulder_angle - math.pi/2 if nose_offset_x > 0 else shoulder_angle + math.pi/2
                    
                    # Determine if facing camera based on target position and nose offset
                    if target_in_front_of_camera:
                        # Target is between user and camera
                        # If nose offset is significant, user might be facing target (away from camera)
                        facing_camera = abs(nose_offset_x) < 0.05
                    else:
                        # Target is behind camera (user is between target and camera)
                        # If nose offset is minimal, user is likely facing camera (and target is behind)
                        facing_camera = abs(nose_offset_x) < 0.05
            else:
                # Fallback to original logic when no target position available
                if abs(nose_offset_x) < 0.05:  # Nose centered - facing camera
                    facing_angle = shoulder_angle + math.pi/2
                    facing_camera = True
                else:
                    facing_angle = shoulder_angle - math.pi/2  
                    facing_camera = False
                
            confidence = min(left_shoulder.visibility, right_shoulder.visibility) * 0.6
        
        # Convert to degrees and normalize to 0-360
        body_angle_degrees = math.degrees(facing_angle)
        if body_angle_degrees < 0:
            body_angle_degrees += 360
        
        # Determine cardinal direction based on facing angle
        if 315 <= body_angle_degrees or body_angle_degrees <= 45:
            primary_direction = "east"
        elif 45 < body_angle_degrees <= 135:
            primary_direction = "south"
        elif 135 < body_angle_degrees <= 225:
            primary_direction = "west"
        else:
            primary_direction = "north"
        
        return {
            'angle': body_angle_degrees,
            'primary_direction': primary_direction,
            'facing_camera': facing_camera,
            'face_visible': face_visible,
            'method_used': 'facial_features' if face_visible else 'shoulder_fallback',
            'confidence': confidence
        }
    
    def calculate_accurate_navigation(self, person_3d, target_3d, user_orientation):
        """
        Calculate accurate navigation vector with proper coordinate transforms
        Everything relative to the target object position and user orientation
        """
        if person_3d is None or target_3d is None:
            return None
            
        # Navigation vector in camera coordinates (from person to target)
        nav_vector = target_3d - person_3d
        
        # Calculate 3D distance to target
        distance_3d = np.linalg.norm(nav_vector)
        
        # Calculate horizontal distance (ignore vertical component)
        horizontal_vector = np.array([nav_vector[0], 0, nav_vector[2]])
        horizontal_distance = np.linalg.norm(horizontal_vector)
        
        # Calculate angle from person to target in camera frame
        target_angle_camera = math.atan2(nav_vector[0], nav_vector[2])  # X over Z
        target_angle_degrees = math.degrees(target_angle_camera)
        
        # Calculate vertical angle (up/down relative to target)
        vertical_angle = math.atan2(-nav_vector[1], horizontal_distance)  # -Y over horizontal distance
        vertical_angle_degrees = math.degrees(vertical_angle)
        
        # Determine user's facing direction relative to target
        facing_target = False
        turn_angle = 0
        relative_target_direction = "unknown"
        
        if user_orientation and user_orientation['confidence'] > 0.5:
            # User's facing angle in radians (corrected orientation)
            user_facing_angle = math.radians(user_orientation['angle'])
            facing_camera = user_orientation.get('facing_camera', True)
            
            # Calculate the angle difference between user facing direction and target direction
            angle_diff = target_angle_camera - user_facing_angle
            
            # Normalize to -π to π
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            turn_angle = math.degrees(angle_diff)
            
            # SIMPLIFIED LOGIC: Better handle facing target determination
            # Focus on 4 simple directions: forward, backward, left, right
            
            if not facing_camera:  # User is facing away from camera
                # When user faces away from camera, use angle-based guidance
                if abs(angle_diff) < math.pi/6:  # Within 30 degrees when facing away
                    facing_target = True
                    relative_target_direction = "straight ahead"
                elif abs(angle_diff) > math.pi*2/3:  # More than 120 degrees
                    facing_target = False
                    relative_target_direction = "behind you"
                else:
                    facing_target = abs(angle_diff) < math.pi/4  # Within 45 degrees
                    if 0 < angle_diff <= math.pi/2:
                        relative_target_direction = "to your left"  # Inverted for backwards
                    elif -math.pi/2 <= angle_diff < 0:
                        relative_target_direction = "to your right"  # Inverted for backwards
                    else:
                        relative_target_direction = "behind and to your side"
            else:
                # IMPROVED SIMPLE LOGIC: When user faces camera (forward)
                # Use Z-coordinate comparison for forward/backward determination
                
                if target_3d is not None and person_3d is not None:
                    # Simple Z-distance comparison
                    target_z = target_3d[2]
                    person_z = person_3d[2]
                    
                    # If target is closer to camera than user, it's in front
                    if target_z < person_z:
                        # Target is between user and camera = in front of user
                        facing_target = abs(angle_diff) < math.pi/2  # Within 90 degrees
                        
                        if abs(angle_diff) < math.pi/12:  # Within 15 degrees
                            relative_target_direction = "straight ahead"
                        elif abs(angle_diff) < math.pi/6:  # Within 30 degrees
                            relative_target_direction = "slightly ahead"
                        elif 0 < angle_diff <= math.pi/2:
                            relative_target_direction = "ahead and to your right"
                        elif -math.pi/2 <= angle_diff < 0:
                            relative_target_direction = "ahead and to your left"
                        else:
                            relative_target_direction = "to your side"
                    else:
                        # Target is farther from camera than user = behind user
                        facing_target = False
                        relative_target_direction = "behind you"
                else:
                    # Fallback to angle-based logic if no 3D data
                    facing_target = abs(angle_diff) < math.pi/2
                    if abs(angle_diff) < math.pi/6:
                        relative_target_direction = "ahead"
                    else:
                        relative_target_direction = "to your side"
        else:
            # Fallback if no orientation data
            turn_angle = target_angle_degrees
            relative_target_direction = "in front"
        
        # Calculate movement components relative to user's current facing direction
        if user_orientation and user_orientation['confidence'] > 0.5:
            user_facing_angle_rad = math.radians(user_orientation['angle'])
            facing_camera = user_orientation.get('facing_camera', True)
            
            # Transform navigation vector to user's coordinate system
            # User's forward direction (where they are facing)
            user_forward = np.array([math.sin(user_facing_angle_rad), 0, math.cos(user_facing_angle_rad)])
            # User's right direction (90 degrees clockwise from forward)
            user_right = np.array([math.cos(user_facing_angle_rad), 0, -math.sin(user_facing_angle_rad)])
            
            # Project navigation vector onto user's coordinate system
            forward_component = np.dot(horizontal_vector, user_forward)
            right_component = np.dot(horizontal_vector, user_right)
            
            # CORRECTION: Adjust components for backwards facing
            if not facing_camera:
                # When facing backwards, right component should be inverted for natural instructions
                right_component = -right_component
        else:
            # Fallback to camera coordinates
            forward_component = nav_vector[2]  # Z component
            right_component = nav_vector[0]   # X component
        
        # Vertical component (always in camera coordinates)
        up_component = -nav_vector[1]  # Y component (inverted because Y is down)
        
        return {
            'vector': nav_vector,
            'distance_3d': distance_3d,
            'horizontal_distance': horizontal_distance,
            'target_angle_camera': target_angle_degrees,
            'vertical_angle': vertical_angle_degrees,
            'facing_target': facing_target,
            'turn_angle': turn_angle,
            'relative_target_direction': relative_target_direction,
            'forward_component': forward_component,
            'right_component': right_component,
            'up_component': up_component,
            'user_facing_camera': user_orientation.get('facing_camera', True) if user_orientation else True,
            'normalized': nav_vector / distance_3d if distance_3d > 0 else np.zeros(3)
        }
    
    def generate_precise_instructions(self, navigation_data, user_orientation):
        """
        Generate precise, actionable navigation instructions relative to target object
        """
        if not navigation_data:
            return "Searching for target..."
            
        distance_3d = navigation_data['distance_3d']
        horizontal_distance = navigation_data['horizontal_distance']
        facing_target = navigation_data['facing_target']
        turn_angle = navigation_data['turn_angle']
        relative_direction = navigation_data['relative_target_direction']
        forward_comp = navigation_data['forward_component']
        right_comp = navigation_data['right_component']
        up_comp = navigation_data['up_component']
        vertical_angle = navigation_data['vertical_angle']
        
        instructions = []
        
        # Distance to target (use horizontal distance for ground movement)
        if horizontal_distance > 200:
            instructions.append(f"[TARGET] {horizontal_distance/100:.1f}m away")
        else:
            instructions.append(f"[TARGET] {horizontal_distance:.0f}cm away")
        
        # Check if we've reached the target - lowered threshold to 15cm
        if horizontal_distance < 15:
            return "[SUCCESS] TARGET REACHED! You're right at the object!"
        elif horizontal_distance < 40:  # Scaled down from 60cm to 40cm
            instructions = [f"[HOT] VERY CLOSE! Target just {horizontal_distance:.0f}cm away"]
            
            # Fine positioning for very close distances
            if abs(right_comp) > 5:  # Reduced from 10cm to 5cm for finer control
                if right_comp > 0:
                    instructions.append("Adjust slightly RIGHT")
                else:
                    instructions.append("Adjust slightly LEFT")
            
            if abs(forward_comp) > 5:  # Reduced from 10cm to 5cm for finer control
                if forward_comp > 0:
                    instructions.append("Step forward a bit")
                else:
                    instructions.append("Step back a bit")
            
            return " | ".join(instructions)
        
        # SIMPLIFIED 4-DIRECTIONAL GUIDANCE
        # Primary orientation guidance - simple 4 directions
        if not facing_target:
            # Simplified turning instructions based on target location
            if "behind you" in relative_direction:
                instructions.append("[TURN] Turn around - target is behind you")
            elif "ahead" in relative_direction:
                instructions.append("[FORWARD] Move forward - target is ahead")
            elif "right" in relative_direction:
                instructions.append("[TURN] Turn RIGHT")
            elif "left" in relative_direction:
                instructions.append("[TURN] Turn LEFT")
            else:
                instructions.append(f"[COMPASS] Target is {relative_direction}")
        else:
            instructions.append("[OK] Facing target")
            
            # Simple movement instructions when facing target
            if horizontal_distance > 80:
                instructions.append("[FORWARD] Move FORWARD")
            else:
                instructions.append("[FORWARD] Move forward slowly")
                
            # Add left/right adjustments if needed
            if abs(turn_angle) > 15:  # Significant side adjustment needed
                if turn_angle > 0:
                    instructions.append("[RIGHT] Bear slightly RIGHT")
                else:
                    instructions.append("[LEFT] Bear slightly LEFT")
        
        # Vertical guidance for objects at different heights
        if abs(vertical_angle) > 15:
            if vertical_angle > 20:
                instructions.append("[LOOK] Look UP - target is above you")
            elif vertical_angle < -20:
                instructions.append("[LOOK] Look DOWN - target is below you")
        
        # Distance-based encouragement
        if horizontal_distance < 100:
            instructions.append("[TARGET] Getting close!")
        
        return " | ".join(instructions)
    
    def draw_enhanced_text(self, frame, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2, bg_color=None):
        """
        Draw enhanced text with background and better visibility
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        
        # Draw background rectangle if specified
        if bg_color is not None:
            cv2.rectangle(frame, 
                         (x - 5, y - text_height - 5), 
                         (x + text_width + 5, y + baseline + 5), 
                         bg_color, -1)
        
        # Draw text with outline for better visibility
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)  # Black outline
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)  # Main text
        
        return text_height + 10  # Return height for next line positioning
    
    def get_status_color(self, status_type, value=None):
        """
        Get appropriate colors for different status types
        """
        colors = {
            'success': (0, 255, 0),      # Green
            'warning': (0, 165, 255),    # Orange
            'error': (0, 0, 255),        # Red
            'info': (255, 255, 0),       # Cyan
            'tracking': (0, 255, 255),   # Yellow
            'target': (255, 0, 255),     # Magenta
            'navigation': (255, 255, 255) # White
        }
        return colors.get(status_type, (255, 255, 255))
    
    def convert_emoji_text(self, text):
        """
        Convert emoji-heavy text to OpenCV-friendly symbols
        """
        replacements = {
            '🎯': '[TARGET]',
            '🔥': '[HOT]',
            '🚶‍♂️': '[WALK]',
            '🚶': '[MOVE]',
            '🧭': '[COMPASS]',
            '🔄': '[TURN]',
            '✅': '[OK]',
            '❌': '[NO]',
            '👀': '[LOOK]',
            '👤': '[PERSON]',
            '🔒': '[LOCK]',
            '🔓': '[UNLOCK]',
            '📷': '[CAM]',
            '🔮': '[PRED]',
            '🔍': '[SEARCH]',
            '🎉': '[SUCCESS]',
            '🎯⬆️': '[AHEAD]',
            '👁️': '[EYE]',
            '🤷': '[BODY]',
            '⬆️': '^',
            '⬇️': 'v',
            '➡️': '>',
            '⬅️': '<'
        }
        
        converted_text = text
        for emoji, replacement in replacements.items():
            converted_text = converted_text.replace(emoji, replacement)
        
        return converted_text
    
    def smooth_position_advanced(self, position, object_type='default'):
        """
        Advanced position smoothing with object-specific filtering
        """
        self.position_filter.append(position)
        
        if len(self.position_filter) < 5:
            return position
        
        positions = np.array(self.position_filter)
        
        # Remove outliers (positions too far from median)
        median_pos = np.median(positions, axis=0)
        distances = np.linalg.norm(positions - median_pos, axis=1)
        threshold = np.percentile(distances, 75)
        
        valid_positions = positions[distances <= threshold]
        
        if len(valid_positions) > 0:
            # Weighted average favoring recent positions
            weights = np.exp(np.linspace(-1, 0, len(valid_positions)))
            weights /= weights.sum()
            smoothed = np.average(valid_positions, weights=weights, axis=0)
            return smoothed
        
        return position
    
    def process_frame(self, frame):
        """
        Main frame processing with enhanced accuracy and robust person tracking
        Focus only on the locked person once established
        """
        # Run YOLO detection
        results = self.model(frame, device=self.device)
        
        # Collect person detections for tracking
        person_detections = []
        target_detections = []
        
        # Process YOLO detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if confidence > self.min_detection_confidence:
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        if class_name == 'person':
                            person_detections.append((bbox, confidence))
                        elif class_name == self.target_item:
                            target_detections.append((bbox, confidence, class_name))
        
        # Update person tracker
        self.person_tracker.update(person_detections)
        
        # Update target tracker
        self.target_tracker.update(target_detections)
        
        # Get the best tracked person
        person_id, person_track = self.person_tracker.get_best_person()
        
        # Get the best tracked target
        target_id, target_track = self.target_tracker.get_best_target()
        
        # Initialize variables
        person_bbox = None
        target_bbox = None
        person_3d = None
        target_3d = None
        user_orientation = None
        pose_results = None
        is_target_predicted = False
        
        # Process ONLY the locked/best tracked person with MediaPipe
        if person_track is not None:
            person_bbox = person_track['bbox']
            
            try:
                # Extract person ROI for more accurate pose estimation
                person_roi, roi_offset = self.extract_person_roi(frame, person_bbox, padding=30)
                
                if person_roi.size > 0:
                    # Run pose estimation only on the person ROI
                    rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(rgb_roi)
                    
                    # Adjust landmarks to full frame coordinates
                    if pose_results.pose_landmarks:
                        pose_results.pose_landmarks = self.adjust_landmarks_to_full_frame(
                            pose_results.pose_landmarks, roi_offset, person_roi.shape, frame.shape
                        )
                    
                    # Enhanced depth estimation for the tracked person
                    depth = self.estimate_depth_multi_method(
                        person_bbox, 'person', pose_results.pose_landmarks, frame.shape
                    )
                    
                    if depth:
                        person_3d_raw = self.get_accurate_3d_position(person_bbox, depth, frame.shape)
                        if person_3d_raw is not None:
                            person_3d = self.smooth_position_advanced(person_3d_raw, 'person')
            except Exception as e:
                print(f"Warning: MediaPipe processing error: {e}")
                # Fallback to basic depth estimation without pose landmarks
                depth = self.estimate_depth_multi_method(person_bbox, 'person', None, frame.shape)
                if depth:
                    person_3d_raw = self.get_accurate_3d_position(person_bbox, depth, frame.shape)
                    if person_3d_raw is not None:
                        person_3d = self.smooth_position_advanced(person_3d_raw, 'person')
        
        # Process target from tracker (includes predicted positions)
        if target_track is not None:
            target_bbox = target_track['bbox']
            is_target_predicted = target_track.get('is_predicted', False)
            
            # Use appropriate class name for depth estimation
            class_name = target_track.get('class_name', self.target_item)
            depth = self.estimate_depth_multi_method(target_bbox, class_name, None, frame.shape)
            
            if depth:
                target_3d_raw = self.get_accurate_3d_position(target_bbox, depth, frame.shape)
                if target_3d_raw is not None:
                    target_3d = self.smooth_position_advanced(target_3d_raw, 'target')
        
        # Enhanced user orientation (only if we have pose results)
        if pose_results and pose_results.pose_landmarks:
            try:
                user_orientation = self.get_robust_user_orientation(
                    pose_results.pose_landmarks, frame.shape, target_3d, person_3d
                )
                
                # Draw pose landmarks ONLY for the locked/tracked person
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                
                # Draw facial feature indicators when using face tracking
                if user_orientation and user_orientation.get('method_used') == 'facial_features':
                    # Draw face landmarks more prominently
                    landmarks = pose_results.pose_landmarks.landmark
                    h, w = frame.shape[:2]
                    
                    # Highlight key facial features
                    nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                    left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
                    right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
                    
                    # Draw enlarged facial feature points
                    cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 255), -1)
                    cv2.circle(frame, (int(left_eye.x * w), int(left_eye.y * h)), 4, (255, 0, 255), -1)
                    cv2.circle(frame, (int(right_eye.x * w), int(right_eye.y * h)), 4, (255, 0, 255), -1)
                    
                    # Draw face direction indicator
                    eye_center_x = int(((left_eye.x + right_eye.x) / 2) * w)
                    eye_center_y = int(((left_eye.y + right_eye.y) / 2) * h)
                    nose_x = int(nose.x * w)
                    nose_y = int(nose.y * h)
                    
                    cv2.arrowedLine(frame, (eye_center_x, eye_center_y), (nose_x, nose_y), 
                                   (0, 255, 255), 3, tipLength=0.3)
            except Exception as e:
                print(f"Warning: Pose processing error: {e}")
                user_orientation = None
        
        # Draw ONLY the locked/best tracked person
        if person_track is not None and person_id is not None:
            bbox = person_track['bbox']
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Color and label for the tracked person
            if person_id == self.person_tracker.locked_person_id:
                color = (0, 255, 0)  # Green for locked person
                thickness = 3
                label = f'Person {person_id} (LOCKED)'
            else:
                color = (255, 255, 0)  # Yellow for best person (not yet locked)
                thickness = 2
                label = f'Person {person_id} (TRACKING)'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced text rendering
            label_clean = self.convert_emoji_text(label)
            self.draw_enhanced_text(frame, label_clean, (x1, y1-10), 
                                  font_scale=0.6, color=color, thickness=2, 
                                  bg_color=(0, 0, 0))
            
            # Show confidence and tracking info
            conf_text = f'Conf: {person_track["confidence"]:.2f}'
            self.draw_enhanced_text(frame, conf_text, (x1, y2+15), 
                                  font_scale=0.4, color=color, thickness=1)
            
            # Show distance
            if person_3d is not None:
                distance = np.linalg.norm(person_3d)
                dist_text = f'Dist: {distance:.1f}cm'
                self.draw_enhanced_text(frame, dist_text, (x1, y2+35), 
                                      font_scale=0.5, color=color, thickness=2)
        
        # Draw target detection (including predicted positions)
        if target_bbox is not None and target_id is not None:
            x1, y1, x2, y2 = target_bbox.astype(int)
            
            # Color and style based on detection vs prediction
            if is_target_predicted:
                color = (0, 165, 255)  # Orange for predicted
                thickness = 2
                style = cv2.LINE_4  # Dashed style
                label = f'{self.target_item.upper()} (PREDICTED)'
                # Draw dashed rectangle for predicted targets
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_4)
            else:
                color = (0, 0, 255)  # Red for actual detection
                thickness = 3
                style = cv2.LINE_8
                label = f'{self.target_item.upper()}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add lock indicator if target is locked
            if target_id == self.target_tracker.locked_target_id:
                label += " [LOCKED]"
            
            # Enhanced text rendering
            label_clean = self.convert_emoji_text(label)
            self.draw_enhanced_text(frame, label_clean, (x1, y1-10), 
                                  font_scale=0.7, color=color, thickness=2, 
                                  bg_color=(0, 0, 0))
            
            # Show tracking info
            if target_track is not None:
                frames_lost = target_track.get('frames_lost', 0)
                if frames_lost > 0:
                    lost_text = f'Lost: {frames_lost} frames'
                    self.draw_enhanced_text(frame, lost_text, (x1, y1-35), 
                                          font_scale=0.4, color=color, thickness=1)
                
                if is_target_predicted:
                    pred_conf = target_track.get('prediction_confidence', 0.0)
                    pred_text = f'Prediction: {pred_conf:.2f}'
                    self.draw_enhanced_text(frame, pred_text, (x1, y2+15), 
                                          font_scale=0.4, color=color, thickness=1)
            
            if target_3d is not None:
                distance_3d = np.linalg.norm(target_3d)
                # Also show horizontal distance for ground navigation
                horizontal_dist = np.linalg.norm([target_3d[0], 0, target_3d[2]])
                
                dist_3d_text = f'3D: {distance_3d:.0f}cm'
                ground_text = f'Ground: {horizontal_dist:.0f}cm'
                
                self.draw_enhanced_text(frame, dist_3d_text, (x1, y2+35), 
                                      font_scale=0.4, color=color, thickness=1)
                self.draw_enhanced_text(frame, ground_text, (x1, y2+50), 
                                      font_scale=0.4, color=color, thickness=1)
        
        # Calculate and display navigation (only for the tracked person)
        navigation_data = None
        if person_3d is not None and target_3d is not None and person_bbox is not None:
            navigation_data = self.calculate_accurate_navigation(
                person_3d, target_3d, user_orientation
            )
            
            # Draw navigation vector with enhanced visibility
            if target_bbox is not None:
                p1 = (int((person_bbox[0] + person_bbox[2]) / 2), 
                      int((person_bbox[1] + person_bbox[3]) / 2))
                p2 = (int((target_bbox[0] + target_bbox[2]) / 2), 
                      int((target_bbox[1] + target_bbox[3]) / 2))
                
                # Draw thick line with outline for better visibility
                cv2.line(frame, p1, p2, (0, 0, 0), 6)  # Black outline
                cv2.line(frame, p1, p2, (255, 0, 255), 4)  # Magenta line
                
                # Enhanced circle markers
                cv2.circle(frame, p1, 12, (0, 0, 0), -1)  # Black outline
                cv2.circle(frame, p1, 10, (0, 255, 0), -1)  # Green person marker
                cv2.circle(frame, p2, 12, (0, 0, 0), -1)  # Black outline
                cv2.circle(frame, p2, 10, (0, 0, 255), -1)  # Red target marker
                
                # Draw angle arc and target information
                if navigation_data:
                    angle = navigation_data['turn_angle']
                    facing = navigation_data['facing_target']
                    
                    # Enhanced facing indicator
                    if navigation_data['facing_target'] and abs(angle) <= 8:
                        facing_status = "[AHEAD]"  # Directly ahead
                    elif navigation_data['facing_target']:
                        facing_status = "[OK]"     # Facing but not centered
                    else:
                        facing_status = "[NO]"     # Not facing
                    
                    angle_text = f'Turn: {angle:.1f}deg {facing_status}'
                    self.draw_enhanced_text(frame, angle_text, (p1[0]-80, p1[1]-20), 
                                          font_scale=0.5, color=(255, 255, 255), thickness=2, 
                                          bg_color=(0, 0, 0))
                    
                    # Show relative direction with enhanced indicators
                    rel_dir = navigation_data['relative_target_direction']
                    if abs(angle) <= 8 and navigation_data['facing_target']:
                        rel_dir = "DIRECTLY AHEAD"
                    
                    rel_dir_clean = self.convert_emoji_text(rel_dir)
                    target_text = f'Target: {rel_dir_clean}'
                    self.draw_enhanced_text(frame, target_text, (p1[0]-80, p1[1]-45), 
                                          font_scale=0.4, color=(255, 165, 0), thickness=2, 
                                          bg_color=(0, 0, 0))
                    
                    # Debug: Show user and target angles
                    if user_orientation:
                        user_angle = user_orientation['angle']
                        target_angle = navigation_data['target_angle_camera']
                        cv2.putText(frame, f'User: {user_angle:.1f}° Target: {target_angle:.1f}°', 
                                   (p1[0]-60, p1[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Generate instructions
        current_time = time.time()
        if current_time - self.last_instruction_time >= self.instruction_cooldown:
            if person_id is not None and target_id is not None:
                # Include prediction info in instructions
                if is_target_predicted:
                    instruction = f"[PRED] Tracking {self.target_item} (predicted) | "
                    instruction += self.generate_precise_instructions(navigation_data, user_orientation)
                else:
                    instruction = self.generate_precise_instructions(navigation_data, user_orientation)
            elif person_id is not None:
                instruction = f"[SEARCH] Person tracked, searching for {self.target_item}..."
            else:
                instruction = "[PERSON] Searching for person to track..."
            self.last_instruction_time = current_time
            self._last_instruction = instruction
        else:
            instruction = getattr(self, '_last_instruction', 'Processing...')
        
        # Display enhanced information with better formatting
        info_y = 30
        
        # Create semi-transparent overlay for better text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (500, info_y + 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Tracking status - show only if we have a person
        if person_id is not None:
            status_text = f"[PERSON] Tracking Person {person_id}"
            if self.person_tracker.locked_person_id == person_id:
                status_text += " [LOCKED]"
            else:
                lock_remaining = self.person_tracker.lock_threshold - self.person_tracker.lock_counter
                status_text += f" (Lock in {lock_remaining})"
            
            info_y += self.draw_enhanced_text(frame, status_text, (10, info_y), 
                                            font_scale=0.6, color=self.get_status_color('success'), 
                                            thickness=2)
        
        # Target tracking status
        if target_id is not None:
            target_status = f"[TARGET] Tracking {self.target_item.title()} {target_id}"
            if self.target_tracker.locked_target_id == target_id:
                target_status += " [LOCKED]"
            else:
                lock_remaining = self.target_tracker.lock_threshold - self.target_tracker.lock_counter
                target_status += f" (Lock in {lock_remaining})"
            
            if is_target_predicted:
                target_status += " [PREDICTED]"
            
            # Color based on detection vs prediction
            color = self.get_status_color('warning') if is_target_predicted else self.get_status_color('target')
            info_y += self.draw_enhanced_text(frame, target_status, (10, info_y), 
                                            font_scale=0.6, color=color, thickness=2)
        
        # User orientation and target relationship
        if user_orientation and user_orientation['confidence'] > 0.5:
            method = "[EYE]" if user_orientation.get('method_used') == 'facial_features' else "[BODY]"
            facing_indicator = "[CAM]" if user_orientation.get('facing_camera') else "[BACK]"
            orientation_text = f"{method} {facing_indicator} | Facing: {user_orientation['primary_direction']} ({user_orientation['angle']:.1f}deg)"
            
            # Color code based on tracking method and confidence
            if user_orientation.get('method_used') == 'facial_features':
                color = self.get_status_color('info')  # Cyan for face tracking
            else:
                color = self.get_status_color('tracking')  # Yellow for body tracking
                
            info_y += self.draw_enhanced_text(frame, orientation_text, (10, info_y), 
                                            font_scale=0.6, color=color, thickness=2)
            
            # Show confidence and method details
            confidence_text = f"Confidence: {user_orientation['confidence']:.2f}"
            info_y += self.draw_enhanced_text(frame, confidence_text, (10, info_y), 
                                            font_scale=0.4, color=color, thickness=1)
            
            # Show target relationship if we have navigation data
            if navigation_data:
                facing_target = "Yes" if navigation_data['facing_target'] else "No"
                rel_direction = navigation_data['relative_target_direction']
                facing_cam = "Forward" if navigation_data.get('user_facing_camera', True) else "Backward"
                target_info = f"Facing target: {facing_target} ({facing_cam}) | Target: {rel_direction}"
                target_info_clean = self.convert_emoji_text(target_info)
                info_y += self.draw_enhanced_text(frame, target_info_clean, (10, info_y), 
                                                font_scale=0.5, color=self.get_status_color('info'), 
                                                thickness=2)
                
                # Enhanced debug information for simplified logic
                if target_3d is not None and person_3d is not None:
                    target_z = target_3d[2]
                    person_z = person_3d[2]
                    target_in_front = target_z < person_z
                    debug_info = f"Simple Logic: Target Z={target_z:.0f}, Person Z={person_z:.0f}, In Front: {target_in_front}"
                    info_y += self.draw_enhanced_text(frame, debug_info, (10, info_y), 
                                                    font_scale=0.4, color=(128, 128, 128), 
                                                    thickness=1)
        
        # Display navigation instructions with enhanced formatting
        if hasattr(self, '_last_instruction'):
            instruction = self.convert_emoji_text(self._last_instruction)
            instruction_lines = instruction.split(' | ')
            
            for i, line in enumerate(instruction_lines):
                # Color code different types of instructions
                if '[TARGET]' in line or '[HOT]' in line:
                    color = self.get_status_color('success')
                elif '[TURN]' in line or '[COMPASS]' in line:
                    color = self.get_status_color('warning')
                elif '[WALK]' in line or '[MOVE]' in line:
                    color = self.get_status_color('info')
                else:
                    color = self.get_status_color('navigation')
                
                info_y += self.draw_enhanced_text(frame, line, (10, info_y), 
                                                font_scale=0.7, color=color, thickness=2, 
                                                bg_color=(50, 50, 50))
        
        return frame
    
    def run(self):
        """
        Main execution loop
        """
        # Try IP camera first
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("IP camera not available, trying local camera...")
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f">> Enhanced Navigation System v4.7 - Simplified 4-Direction Logic")
        print(f"Target: {self.target_item}")
        print(f"Device: {self.device}")
        print("Features: Simple Forward/Backward/Left/Right, Z-coordinate logic, Clear instructions")
        print("Controls: 'q' quit, 'c' change target, 'r' reset, 'l' reset person lock, 't' reset target lock")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Robust Object Navigation System', processed_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.change_target()
            elif key == ord('r'):
                self.reset_tracking()
            elif key == ord('l'):
                self.person_tracker.reset_lock()
            elif key == ord('t'):
                self.target_tracker.reset_lock()
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        self.hands.close()
    
    def change_target(self):
        """Change target item"""
        items = list(self.object_dimensions.keys())
        print(f"\nAvailable items: {items}")
        new_item = input("Enter new target: ").strip().lower()
        
        if new_item in items:
            self.target_item = new_item
            print(f"Target changed to: {self.target_item}")
        else:
            print("Invalid item.")
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.position_filter.clear()
        self.orientation_filter.clear()
        
        # Clear all object-specific depth filters
        for filter_key in self.depth_filters:
            self.depth_filters[filter_key].clear()
            
        self.navigation_history.clear()
        self.person_tracker.reset_lock()
        self.person_tracker.tracked_persons.clear()
        self.person_tracker.next_id = 0
        self.target_tracker.reset_lock()
        self.target_tracker.tracked_targets.clear()
        self.target_tracker.next_id = 0
        print("All tracking data reset")

# Usage
if __name__ == "__main__":
    try:
        nav_system = RobustObjectNavigationSystem(target_item='cell phone')
        nav_system.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Install: pip install ultralytics opencv-python mediapipe torch scipy")