import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import time
import pickle

class CelebrityChallengeUltimate:
    def __init__(self):
        self.celebs_dir = "celebs"
        self.cache_file = "encodings.pickle"
        if not os.path.exists(self.celebs_dir): os.makedirs(self.celebs_dir)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
        self.connections = self.mp_face_mesh.FACEMESH_TESSELATION
        
        self.celeb_data = []
        self.current_match = None
        
        self.is_registering = False
        self.user_input_name = ""
        self.success_message = ""
        self.success_time = 0
        
        self.sync_database()

    def get_norm_lms(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            coords = np.array([(l.x, l.y) for l in res.multi_face_landmarks[0].landmark])
            return coords - coords.mean(axis=0)
        return None

    def sync_database(self):
        existing_cache = []
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    existing_cache = pickle.load(f)
            except: existing_cache = []

        cached_filenames = {c.get('filename'): c for c in existing_cache if 'filename' in c}
        current_files = [f for f in os.listdir(self.celebs_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        updated_data = []
        new_encodes = 0
        
        print("\n--- syncing database ---")
        for filename in current_files:
            if filename in cached_filenames:
                updated_data.append(cached_filenames[filename])
            else:
                print(f"encoding: {filename}...")
                path = os.path.join(self.celebs_dir, filename)
                img = cv2.imread(path)
                if img is None: continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                if encs:
                    lms = self.get_norm_lms(img)
                    if lms is not None:
                        name_part = filename.rsplit('_', 1)[0]
                        name = name_part.replace('_', ' ').title()
                        updated_data.append({
                            "name": name, 
                            "enc": encs[0], 
                            "lms": lms, 
                            "img": cv2.resize(img, (200, 200)), 
                            "filename": filename
                        })
                        new_encodes += 1
                else: os.remove(path)

        self.celeb_data = updated_data
        if new_encodes > 0 or len(updated_data) != len(existing_cache):
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.celeb_data, f)
            print(f"cache updated. total: {len(self.celeb_data)}")

    def run(self):
        cap = cv2.VideoCapture(0)
        live_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        window_name = 'Celebrity Lookalike'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            cam_h, cam_w = frame.shape[:2]
            
            rect = cv2.getWindowImageRect(window_name)
            win_w, win_h = rect[2], rect[3]
            if win_w < 100: win_w, win_h = 1280, 720 

            sidebar_w = int(win_w * 0.22)
            main_w = win_w - sidebar_w
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

            scale = min(main_w/cam_w, win_h/cam_h)
            new_w, new_h = int(cam_w * scale), int(cam_h * scale)
            feed_resized = cv2.resize(frame, (new_w, new_h))
            x_off, y_off = (main_w - new_w) // 2, (win_h - new_h) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = feed_resized
            canvas[:, main_w:] = (30, 30, 30)

            rgb_live = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = live_mesh.process(rgb_live)

            if self.is_registering:
                cv2.rectangle(canvas, (main_w//2-250, win_h//2-60), (main_w//2+250, win_h//2+60), (0,0,0), -1)
                cv2.rectangle(canvas, (main_w//2-250, win_h//2-60), (main_w//2+250, win_h//2+60), (0,165,255), 2)
                cv2.putText(canvas, "TYPE YOUR NAME:", (main_w//2-230, win_h//2-20), 1, 1.5, (255,255,255), 2)
                cv2.putText(canvas, f"{self.user_input_name}_", (main_w//2-230, win_h//2+30), 1, 2, (0,165,255), 3)
                cv2.putText(canvas, "Press ENTER to save | ESC to cancel", (main_w//2-230, win_h//2+80), 1, 1, (150,150,150), 1)
            else:
                if res.multi_face_landmarks and self.celeb_data:
                    small_rgb = cv2.resize(rgb_live, (0,0), fx=0.25, fy=0.25)
                    face_encs = face_recognition.face_encodings(small_rgb)
                    if face_encs:
                        dists = face_recognition.face_distance([c['enc'] for c in self.celeb_data], face_encs[0])
                        idx = np.argmin(dists)
                        self.current_match = self.celeb_data[idx]
                        
                        side_img_size = sidebar_w - 40
                        if side_img_size > 20:
                            canvas[20:20+side_img_size, main_w+20:main_w+20+side_img_size] = cv2.resize(self.current_match['img'], (side_img_size, side_img_size))
                        
                        f_scale = sidebar_w / 350
                        cv2.putText(canvas, self.current_match['name'], (main_w+20, side_img_size+60), 1, f_scale, (255,255,255), 2)
                        sim = max(0, (1 - dists[idx]) * 100)
                        cv2.putText(canvas, f"{sim:.1f}%", (main_w+20, side_img_size+110), 1, f_scale*2, (0,255,0), 3)
                        
                        lm_list = res.multi_face_landmarks[0].landmark
                        curr_norm = np.array([(l.x, l.y) for l in lm_list]) - np.array([(l.x, l.y) for l in lm_list]).mean(axis=0)
                        errors = np.linalg.norm(curr_norm - self.current_match['lms'], axis=1)

                        for conn in self.connections:
                            avg_err = (errors[conn[0]] + errors[conn[1]]) / 2
                            g, r = max(0, 255-int(avg_err*8500)), min(255, int(avg_err*8500))
                            pt1 = (int(lm_list[conn[0]].x * new_w) + x_off, int(lm_list[conn[0]].y * new_h) + y_off)
                            pt2 = (int(lm_list[conn[1]].x * new_w) + x_off, int(lm_list[conn[1]].y * new_h) + y_off)
                            cv2.line(canvas, pt1, pt2, (0, g, r), 1)

                if time.time() - self.success_time < 3:
                    cv2.putText(canvas, self.success_message, (20, 50), 1, 2, (0,255,0), 3)
                else:
                    cv2.putText(canvas, "Press 'C' to Register | 'Q' to Quit", (20, win_h-20), 1, 1, (180,180,180), 1)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            
            if self.is_registering:
                if key == 13: # ENTER
                    if self.user_input_name.strip():
                        filename = f"{self.user_input_name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
                        cv2.imwrite(os.path.join(self.celebs_dir, filename), frame)
                        self.sync_database()
                        self.success_message = f"SAVED: {self.user_input_name.upper()}"
                        self.success_time = time.time()
                        self.is_registering = False
                        self.user_input_name = ""
                elif key == 27: # ESC
                    self.is_registering = False
                    self.user_input_name = ""
                elif key == 8: # BACKSPACE
                    self.user_input_name = self.user_input_name[:-1]
                elif 32 <= key <= 126:
                    self.user_input_name += chr(key)
            else:
                if key == ord('c'): self.is_registering = True
                elif key == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CelebrityChallengeUltimate().run()








