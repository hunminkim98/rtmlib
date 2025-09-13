import numpy as np
import matplotlib.pyplot as plt

from rtmlib.visualization.skeleton.coco133 import coco133


# prepare mapping and edge list from coco133 for 3D visualization
KP_NAME_TO_ID = {v['name']: v['id'] for v in coco133['keypoint_info'].values()}
SKELETON_EDGES = []
for v in coco133['skeleton_info'].values():
    a_name, b_name = v['link']
    if a_name in KP_NAME_TO_ID and b_name in KP_NAME_TO_ID:
        a_id = KP_NAME_TO_ID[a_name]
        b_id = KP_NAME_TO_ID[b_name]
        color = tuple(c / 255.0 for c in v.get('color', [1, 1, 1]))
        SKELETON_EDGES.append((a_id, b_id, color))


class Visualizer3D:
    """Enhanced 3D visualizer supporting multiple persons with color coding.

    Usage:
        viz = Visualizer3D(multi_person=True)
        viz.update_multi_person(keypoints_dict, scores_dict, track_ids, kpt_thr=0.5)
        viz.close()
        
        # Single person mode (backward compatibility):
        viz = Visualizer3D(person=0)
        viz.update(keypoints, scores, kpt_thr=0.5)
        viz.close()
    """
    
    # Predefined colors for different persons
    PERSON_COLORS = [
        (1.0, 0.0, 0.0),    # Red
        (0.0, 1.0, 0.0),    # Green  
        (0.0, 0.0, 1.0),    # Blue
        (1.0, 1.0, 0.0),    # Yellow
        (1.0, 0.0, 1.0),    # Magenta
        (0.0, 1.0, 1.0),    # Cyan
        (1.0, 0.5, 0.0),    # Orange
        (0.5, 0.0, 1.0),    # Purple
        (0.0, 0.5, 0.0),    # Dark Green
        (0.5, 0.5, 0.5),    # Gray
    ]
    
    def __init__(self, person=0, joint_labels=None, multi_person=False):
        self.person = person
        self.joint_labels = joint_labels
        self.multi_person = multi_person
        self.active_persons = {}  # track_id -> last_seen_frame
        self.frame_count = 0
        
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8) if multi_person else (6, 6))
        try:
            window_title = 'Multi-Person 3D Keypoints' if multi_person else '3D Keypoints'
            self.fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update(self, kpts, scores=None, kpt_thr=0.3):
        if kpts is None:
            k = None
        else:
            k = np.asarray(kpts)
            if k.ndim == 2:
                k = k[np.newaxis, ...]
        if k is None or k.shape[-1] < 3:
            # Clear visualization if no valid keypoints
            self.ax.clear()
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title(f'3D keypoints (person {self.person}) - No detection')
            try:
                self.fig.canvas.draw()
                plt.pause(0.001)
            except Exception:
                pass
            return
        pts = k[self.person, :, :3]

        # normalize scores
        s_arr = None
        if scores is not None:
            s = np.asarray(scores)
            if s.ndim == 1:
                s = s[np.newaxis, ...]
            s_arr = s

        self.ax.clear()
        # draw joints
        if s_arr is None:
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r')
        else:
            person_scores = s_arr[self.person]
            visible = person_scores >= kpt_thr
            for i in range(pts.shape[0]):
                if not visible[i]:
                    continue
                x, y, z = pts[i]
                # ensure alpha is within matplotlib valid range [0, 1]
                alpha = float(np.clip(person_scores[i], 0.0, 1.0))
                self.ax.scatter([x], [y], [z], c='r', alpha=alpha)

        # draw skeleton edges
        for a_id, b_id, color in SKELETON_EDGES:
            if a_id < pts.shape[0] and b_id < pts.shape[0]:
                if s_arr is not None:
                    person_scores = s_arr[self.person]
                    if person_scores[a_id] < kpt_thr or person_scores[b_id] < kpt_thr:
                        continue
                try:
                    x_vals = [pts[a_id, 0], pts[b_id, 0]]
                    y_vals = [pts[a_id, 1], pts[b_id, 1]]
                    z_vals = [pts[a_id, 2], pts[b_id, 2]]
                    self.ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2)
                except Exception:
                    self.ax.plot(x_vals, y_vals, z_vals, color='k', linewidth=2)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'3D keypoints (person {self.person})')
        try:
            self.fig.canvas.draw()
            plt.pause(0.001)
        except Exception:
            pass
    
    def get_person_color(self, track_id):
        """Get color for a specific person based on track ID."""
        color_idx = track_id % len(self.PERSON_COLORS)
        return self.PERSON_COLORS[color_idx]
    
    def update_multi_person(self, keypoints_list, scores_list=None, track_ids=None, kpt_thr=0.3):
        """Update visualization with multiple persons.
        
        Args:
            keypoints_list: List or array of keypoints for each person (N, J, 3)
            scores_list: List or array of scores for each person (N, J) 
            track_ids: List of track IDs for each person
            kpt_thr: Confidence threshold for displaying keypoints
        """
        self.frame_count += 1
        
        if keypoints_list is None or len(keypoints_list) == 0:
            # Clear visualization if no persons detected
            self.ax.clear()
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')  
            self.ax.set_zlabel('Z')
            self.ax.set_title('Multi-Person 3D Keypoints (No persons detected)')
            try:
                self.fig.canvas.draw()
                plt.pause(0.001)
            except Exception:
                pass
            return
        
        if keypoints_list is None:
            k = None
        else:
            k = np.asarray(keypoints_list)
            if k.ndim == 2:
                k = k[np.newaxis, ...]
        if k is None or k.shape[-1] < 3:
            return
            
        # Default track IDs if not provided
        if track_ids is None:
            track_ids = list(range(len(k)))
            
        # Update active persons tracking
        for track_id in track_ids:
            self.active_persons[track_id] = self.frame_count
            
        # Clear the plot
        self.ax.clear()
        
        # Visualize each person
        legend_elements = []
        for person_idx, track_id in enumerate(track_ids):
            if person_idx >= len(k):
                continue
                
            pts = k[person_idx, :, :3]
            person_color = self.get_person_color(track_id)
            
            # Handle scores for this person
            person_scores = None
            if scores_list is not None:
                s = np.asarray(scores_list)
                if s.ndim == 1:
                    s = s[np.newaxis, ...]
                if person_idx < len(s):
                    person_scores = s[person_idx]
                    
            # Draw joints for this person
            if person_scores is None:
                self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                              c=[person_color], s=30, alpha=0.8, 
                              label=f'Person {track_id}')
            else:
                visible = person_scores >= kpt_thr
                for i in range(pts.shape[0]):
                    if not visible[i]:
                        continue
                    x, y, z = pts[i]
                    alpha = float(np.clip(person_scores[i], 0.3, 1.0))
                    self.ax.scatter([x], [y], [z], c=[person_color], s=30, alpha=alpha)
                        
            # Draw skeleton edges for this person
            for a_id, b_id, _ in SKELETON_EDGES:  # Use person color instead of edge color
                if a_id < pts.shape[0] and b_id < pts.shape[0]:
                    if person_scores is not None:
                        if person_scores[a_id] < kpt_thr or person_scores[b_id] < kpt_thr:
                            continue
                    try:
                        x_vals = [pts[a_id, 0], pts[b_id, 0]]
                        y_vals = [pts[a_id, 1], pts[b_id, 1]] 
                        z_vals = [pts[a_id, 2], pts[b_id, 2]]
                        self.ax.plot(x_vals, y_vals, z_vals, color=person_color, 
                                   linewidth=2, alpha=0.7)
                    except Exception:
                        self.ax.plot(x_vals, y_vals, z_vals, color=person_color,
                                   linewidth=2, alpha=0.7)
                                   
            # Add to legend
            legend_elements.append(f'Person {track_id}')
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Multi-Person 3D Keypoints ({len(track_ids)} persons)')
        
        # Add legend if multiple persons
        if len(track_ids) > 1:
            # Create proxy artists for legend
            legend_handles = []
            for i, track_id in enumerate(track_ids):
                color = self.get_person_color(track_id)
                handle = plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=8, 
                                  label=f'Person {track_id}')
                legend_handles.append(handle)
            self.ax.legend(handles=legend_handles, loc='upper left', 
                          bbox_to_anchor=(0.02, 0.98))
        
        try:
            self.fig.canvas.draw()
            plt.pause(0.001)
        except Exception:
            pass

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass
