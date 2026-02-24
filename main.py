import cv2
import mediapipe as mp
import numpy as np
import math
from threading import Thread

class FastWebcam:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True

class PurePenAR:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0, 
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_style = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

        # --- هنا تم تكبير حجم المربع ---
        self.voxel_size = 45 
        
        self.voxels = set()
        self.camera_z = 800
        self.fov = 800

        self.angle_x = 0.0
        self.angle_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.is_rotating = False
        self.last_rot_x = 0
        self.last_rot_y = 0

        self.pen_smooth_x = None
        self.pen_smooth_y = None
        self.last_raw_x = 0
        self.last_raw_y = 0

        # ألوان القلم والزجاج
        self.color_glass = (255, 255, 0)     
        self.color_edges = (255, 255, 100)   
        self.color_laser = (255, 50, 255)

    def rotate_point(self, x, y, z, ax, ay):
        y1 = y * math.cos(ax) - z * math.sin(ax)
        z1 = y * math.sin(ax) + z * math.cos(ax)
        x2 = x * math.cos(ay) - z1 * math.sin(ay)
        z2 = x * math.sin(ay) + z1 * math.cos(ay)
        return x2, y1, z2

    def inverse_rotate_point(self, x, y, z, ax, ay):
        x1 = x * math.cos(-ay) - z * math.sin(-ay)
        z1 = x * math.sin(-ay) + z * math.cos(-ay)
        y2 = y * math.cos(-ax) - z1 * math.sin(-ax)
        z2 = y * math.sin(-ax) + z1 * math.cos(-ax)
        return x1, y2, z2

    def project(self, x, y, z, cx, cy):
        z_adj = z + self.camera_z
        if z_adj < 1: z_adj = 1
        u = int((x * self.fov) / z_adj) + cx
        v = int((y * self.fov) / z_adj) + cy
        return u, v

    def run(self):
        cam = FastWebcam(src=0, width=1280, height=720).start()

        while True:
            success, frame = cam.read()
            if not success or frame is None: continue
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            cx, cy = w // 2, h // 2

            if not self.is_rotating:
                self.angle_y += self.vel_y
                self.angle_x += self.vel_x
                self.vel_y *= 0.85 
                self.vel_x *= 0.85

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(rgb_frame, (640, 360))
            results = self.hands.process(small_frame)

            left_hand = None
            right_hand = None

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS, self.hand_style)
                    if hand_lms.landmark[9].x < 0.5:
                        left_hand = hand_lms
                    else:
                        right_hand = hand_lms

            # الإيد الشمال (دوران حر)
            if left_hand:
                idx = left_hand.landmark[8]
                thumb = left_hand.landmark[4]
                pinch_dist = math.hypot(idx.x - thumb.x, idx.y - thumb.y)
                screen_x, screen_y = int((idx.x + thumb.x)/2 * w), int((idx.y + thumb.y)/2 * h)

                if pinch_dist < 0.05: 
                    if not self.is_rotating:
                        self.is_rotating = True
                        self.last_rot_x, self.last_rot_y = screen_x, screen_y
                        self.vel_x, self.vel_y = 0, 0
                    else:
                        dx = screen_x - self.last_rot_x
                        dy = screen_y - self.last_rot_y
                        self.angle_y += dx * 0.006 
                        self.angle_x -= dy * 0.006 
                        self.vel_y = dx * 0.006
                        self.vel_x = -dy * 0.006
                        self.last_rot_x, self.last_rot_y = screen_x, screen_y
                else:
                    self.is_rotating = False
            else:
                self.is_rotating = False

            if right_hand:
                idx = right_hand.landmark[8]
                mid = right_hand.landmark[12]
                thumb = right_hand.landmark[4]
                
                raw_x, raw_y = (idx.x + thumb.x)/2 * w, (idx.y + thumb.y)/2 * h
                
                speed = math.hypot(raw_x - self.last_raw_x, raw_y - self.last_raw_y)
                self.last_raw_x, self.last_raw_y = raw_x, raw_y
                dynamic_alpha = np.interp(speed, [0, 40], [0.15, 0.85])

                if self.pen_smooth_x is None:
                    self.pen_smooth_x, self.pen_smooth_y = raw_x, raw_y
                else:
                    self.pen_smooth_x = dynamic_alpha * raw_x + (1 - dynamic_alpha) * self.pen_smooth_x
                    self.pen_smooth_y = dynamic_alpha * raw_y + (1 - dynamic_alpha) * self.pen_smooth_y
                
                wx, wy, wz = self.inverse_rotate_point(self.pen_smooth_x - cx, self.pen_smooth_y - cy, 0, self.angle_x, self.angle_y)
                gx = round(wx / self.voxel_size) * self.voxel_size
                gy = round(wy / self.voxel_size) * self.voxel_size
                gz = round(wz / self.voxel_size) * self.voxel_size
                target_pos = (gx, gy, gz)

                cursor_rx, cursor_ry, cursor_rz = self.rotate_point(*target_pos, self.angle_x, self.angle_y)
                cursor_px, cursor_py = self.project(cursor_rx, cursor_ry, cursor_rz, cx, cy)
                finger_px, finger_py = int(idx.x * w), int(idx.y * h)
                
                cv2.line(frame, (finger_px, finger_py), (cursor_px, cursor_py), self.color_laser, 2, cv2.LINE_AA)
                cv2.circle(frame, (cursor_px, cursor_py), 8, self.color_glass, 2)

                build_dist = math.hypot(idx.x - thumb.x, idx.y - thumb.y)
                if build_dist < 0.04: 
                    self.voxels.add(target_pos)
                    cv2.circle(frame, (finger_px, finger_py), 15, self.color_edges, cv2.FILLED)
                    cv2.circle(frame, (cursor_px, cursor_py), 8, self.color_glass, cv2.FILLED)

                erase_dist = math.hypot(mid.x - thumb.x, mid.y - thumb.y)
                if erase_dist < 0.04: 
                    if target_pos in self.voxels:
                        self.voxels.remove(target_pos)
                    cv2.circle(frame, (finger_px, finger_py), 15, (0, 0, 255), cv2.FILLED)
            else:
                self.pen_smooth_x = None

            rendered_voxels = []
            for vx, vy, vz in self.voxels:
                _, _, rz = self.rotate_point(vx, vy, vz, self.angle_x, self.angle_y)
                rendered_voxels.append((rz, vx, vy, vz))

            rendered_voxels.sort(key=lambda item: item[0], reverse=True)

            glass_overlay = frame.copy()
            edges_list = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            s = self.voxel_size / 2

            for rz, vx, vy, vz in rendered_voxels:
                vertices = [(-s,-s,-s),(s,-s,-s),(s,s,-s),(-s,s,-s),(-s,-s,s),(s,-s,s),(s,s,s),(-s,s,s)]
                pts_2d = []
                for vvx, vvy, vvz in vertices:
                    px, py, pz = self.rotate_point(vx+vvx, vy+vvy, vz+vvz, self.angle_x, self.angle_y)
                    pts_2d.append(self.project(px, py, pz, cx, cy))
                
                hull = cv2.convexHull(np.array(pts_2d))
                cv2.fillConvexPoly(glass_overlay, hull, self.color_glass)
                
                for p1, p2 in edges_list: 
                    cv2.line(frame, pts_2d[p1], pts_2d[p2], self.color_edges, 1, cv2.LINE_AA)
                

            cv2.addWeighted(glass_overlay, 0.15, frame, 0.85, 0, frame)
            
            cv2.putText(frame, f"PURE PEN AR | SIZE: {self.voxel_size}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.color_glass, 1)

            cv2.imshow("Pure Pen AR Studio", frame)

            key = cv2.waitKey(1)
            if key == 27: break
            elif key == ord('c'): 
                self.voxels.clear()
                self.angle_x, self.angle_y = 0.0, 0.0

        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PurePenAR()
    app.run()