#!/usr/bin/env python3
# celebrity face matching thing

from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
try:
    import face_recognition
except ImportError as e:
    print(f"face_recognition import failed: {e}")
except Exception as e:
    print(f"face_recognition error: {e}")
import numpy as np
import os
import pickle
from PIL import Image
import base64
import io
import threading
import time
import requests
from ddgs import DDGS
import random
import json

app = Flask(__name__)

matcher = None
_db_version = 0
_db_version_lock = threading.Lock()


class CelebrityMatcher:
    def __init__(self, celebs_dir="celebs", cache_file="encodings.pickle"):
        self.celebs_dir = celebs_dir
        self.cache_file = cache_file
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            refine_landmarks=True, 
            max_num_faces=1
        )
        self.connections = self.mp_face_mesh.FACEMESH_TESSELATION
        self.celeb_data = []
    
    def get_norm_lms(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            coords = np.array([(l.x, l.y) for l in res.multi_face_landmarks[0].landmark])
            return coords - coords.mean(axis=0)
        return None
    
    def load_database(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if data and len(data) > 0:
                        self.celeb_data = data
                        print(f"loaded {len(self.celeb_data)} celebs from cache", flush=True)
                        return self.celeb_data
                    else:
                        print("cache empty, rebuilding...", flush=True)
            except Exception as e:
                print(f"error loading cache: {e}, rebuilding...")
        
        self.celeb_data = []
        
        preload_dir = "preload"
        all_image_files = []
        
        if os.path.exists(preload_dir):
            preload_files = os.listdir(preload_dir)
            preload_images = [f for f in preload_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in preload_images:
                all_image_files.append(os.path.join(preload_dir, img))
            print(f"found {len(preload_images)} image files in preload dir", flush=True)
        
        if not os.path.exists(self.celebs_dir):
            os.makedirs(self.celebs_dir, exist_ok=True)
        else:
            files = os.listdir(self.celebs_dir)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in image_files:
                all_image_files.append(os.path.join(self.celebs_dir, img))
            print(f"found {len(image_files)} image files in celebs dir", flush=True)
        
        if len(all_image_files) == 0:
            print("no image files found in preload or celebs dirs", flush=True)
            return self.celeb_data
        
        print(f"total {len(all_image_files)} image files to process", flush=True)
        
        for filepath in all_image_files:
            filename = os.path.basename(filepath)
            try:
                rgb = None
                try:
                    rgb = face_recognition.load_image_file(filepath)
                except:
                    try:
                        pil_img = Image.open(filepath)
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        rgb = np.array(pil_img, dtype=np.uint8)
                    except Exception as e:
                        print(f"failed to load {filename}: {e}", flush=True)
                        continue
                
                if rgb is None or len(rgb.shape) != 3 or rgb.shape[2] != 3:
                    print(f"invalid image format for {filename}: shape {rgb.shape if rgb is not None else 'None'}", flush=True)
                    continue
                
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                h, w = rgb.shape[:2]
                if h > 800 or w > 800:
                    scale = min(800.0 / h, 800.0 / w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    rgb = cv2.resize(rgb, (new_w, new_h))
                    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                try:
                    face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
                except Exception as e:
                    print(f"error encoding {filename}: {e}, shape={rgb.shape}, dtype={rgb.dtype}", flush=True)
                    continue
                
                if face_encs and len(face_encs) > 0:
                    name = os.path.splitext(filename)[0]
                    lms = self.get_norm_lms(img_bgr)
                    if lms is None:
                        lms = np.zeros((468, 2))
                    
                    self.celeb_data.append({
                        'name': name,
                        'enc': face_encs[0],
                        'lms': lms,
                        'img_path': filepath
                    })
                    print(f"encoded {filename}", flush=True)
                else:
                    print(f"no face found in {filename}", flush=True)
            except Exception as e:
                print(f"error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"database loaded: {len(self.celeb_data)} celebrities", flush=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.celeb_data, f)
        
        return self.celeb_data
    
    def find_match(self, frame_rgb):
        if not self.celeb_data or len(self.celeb_data) == 0:
            print(f"find_match: no celeb data, count={len(self.celeb_data) if self.celeb_data else 0}", flush=True)
            return None, None, None
        
        if len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
            return None, None, None
        
        h, w = frame_rgb.shape[:2]
        new_w = int(w * 0.25)
        new_h = int(h * 0.25)
        small_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        small_rgb = np.ascontiguousarray(small_rgb, dtype=np.uint8)
        
        if small_rgb.dtype != np.uint8:
            small_rgb = small_rgb.astype(np.uint8)
        
        if small_rgb.min() < 0 or small_rgb.max() > 255:
            small_rgb = np.clip(small_rgb, 0, 255).astype(np.uint8)
        
        if not small_rgb.flags['C_CONTIGUOUS']:
            small_rgb = np.ascontiguousarray(small_rgb, dtype=np.uint8)
        
        try:
            face_encs = face_recognition.face_encodings(small_rgb, num_jitters=1)
        except Exception as e:
            print(f"find_match: face_encodings error: {e}", flush=True)
            return None, None, None
        
        if not face_encs or len(face_encs) == 0:
            print(f"find_match: no face detected in frame", flush=True)
            return None, None, None
        
        celeb_encs = [c['enc'] for c in self.celeb_data]
        dists = face_recognition.face_distance(celeb_encs, face_encs[0])
        idx = np.argmin(dists)
        distance = dists[idx]
        similarity = max(0, (1 - distance) * 100)
        
        celeb_names = [c['name'] for c in self.celeb_data]
        print(f"find_match: checking {len(self.celeb_data)} celebs: {celeb_names}", flush=True)
        print(f"find_match: best match: {self.celeb_data[idx]['name']} with distance={distance:.4f}, similarity={similarity:.2f}%", flush=True)
        
        if distance > 0.6:
            print(f"find_match: distance high ({distance:.4f} > 0.6), but returning best match anyway", flush=True)
        
        print(f"find_match: matched {self.celeb_data[idx]['name']} with similarity {similarity:.2f}%", flush=True)
        
        return self.celeb_data[idx], distance, similarity


import sys
sys.stdout.flush()
print("initializing matcher...", flush=True)
try:
    matcher = CelebrityMatcher()
    matcher.load_database()
    celeb_count = len(matcher.celeb_data)
    print(f"loaded {celeb_count} celebrities", flush=True)
    
    if celeb_count == 0:
        print("database empty, seeding in background...", flush=True)
        def seed_background():
            try:
                from seed_celebs import seed_celebs
                print("starting seed process...", flush=True)
                added = seed_celebs()
                print(f"seed finished, added {added} images", flush=True)
                if matcher:
                    matcher.load_database()
                    print(f"database reloaded. total: {len(matcher.celeb_data)}", flush=True)
            except Exception as e:
                print(f"seed failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=seed_background, daemon=True).start()
except Exception as e:
    print(f"matcher init failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    matcher = None


def process_frame(img_array, matcher):
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return None, None, None
    
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)
    
    if rgb_img.dtype != np.uint8:
        rgb_img = rgb_img.astype(np.uint8)
    
    if rgb_img.min() < 0 or rgb_img.max() > 255:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        max_num_faces=1
    )
    
    results = face_mesh.process(rgb_img)
    
    match = None
    similarity = 0.0
    
    if results.multi_face_landmarks:
        if matcher and matcher.celeb_data and len(matcher.celeb_data) > 0:
            match, distance, similarity = matcher.find_match(rgb_img)
        
        lm_list = results.multi_face_landmarks[0].landmark
        connections = mp_face_mesh.FACEMESH_TESSELATION
        
        if match:
            curr_lms = np.array([(l.x, l.y) for l in lm_list])
            curr_norm = curr_lms - curr_lms.mean(axis=0)
            errors = np.linalg.norm(curr_norm - match['lms'], axis=1)
            for conn in connections:
                err1 = errors[conn[0]]
                err2 = errors[conn[1]]
                avg_err = (err1 + err2) / 2
                g = max(0, 255 - int(avg_err * 8500))
                r = min(255, int(avg_err * 8500))
                
                x1 = int(lm_list[conn[0]].x * w)
                y1 = int(lm_list[conn[0]].y * h)
                x2 = int(lm_list[conn[1]].x * w)
                y2 = int(lm_list[conn[1]].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, g, r), 1)
        else:
            for conn in connections:
                x1 = int(lm_list[conn[0]].x * w)
                y1 = int(lm_list[conn[0]].y * h)
                x2 = int(lm_list[conn[1]].x * w)
                y2 = int(lm_list[conn[1]].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), match, similarity


def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64.decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    celeb_count = len(matcher.celeb_data) if matcher else 0
    celebs_dir = "celebs"
    file_count = 0
    if os.path.exists(celebs_dir):
        file_count = len([f for f in os.listdir(celebs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    return jsonify({
        'celeb_count': celeb_count,
        'file_count': file_count,
        'matcher_initialized': matcher is not None
    })


@app.route('/api/reload')
def reload_database():
    if matcher:
        matcher.load_database()
        return jsonify({
            'success': True,
            'count': len(matcher.celeb_data),
            'message': f'reloaded {len(matcher.celeb_data)} celebrities'
        })
    return jsonify({'error': 'matcher not initialized'}), 500


@app.route('/api/seed', methods=['POST'])
def trigger_seed():
    if not matcher:
        return jsonify({'error': 'matcher not initialized'}), 500
    
    def seed_background():
        try:
            from seed_celebs import seed_celebs
            print("manual seed triggered...")
            added = seed_celebs()
            print(f"seed finished, added {added} images")
            if matcher:
                matcher.load_database()
                print(f"database reloaded. total: {len(matcher.celeb_data)}")
        except Exception as e:
            print(f"seed failed: {e}")
            import traceback
            traceback.print_exc()
    
    threading.Thread(target=seed_background, daemon=True).start()
    return jsonify({'success': True, 'message': 'seeding started in background'})


@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'no image selected'}), 400
        
        image = Image.open(io.BytesIO(file.read()))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image, dtype=np.uint8)
        
        if not matcher:
            return jsonify({'error': 'matcher not initialized'}), 500
        
        if matcher.celeb_data and len(matcher.celeb_data) == 0:
            matcher.load_database()
        
        try:
            processed_img, match, similarity = process_frame(image_array, matcher)
            if processed_img is None:
                return jsonify({'error': 'failed to process image'}), 500
        except Exception as e:
            print(f"error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'error processing image: {str(e)}'}), 500
        
        if not match:
            try:
                processed_img_str = image_to_base64(processed_img)
                celeb_count = len(matcher.celeb_data) if matcher else 0
                match_name = 'No match found' if celeb_count > 0 else 'No celebrities yet'
                return jsonify({
                    'success': True,
                    'match_name': match_name,
                    'similarity': 0.0,
                    'processed_image': processed_img_str,
                    'celebrity_image': processed_img_str,
                    'face_count': celeb_count
                })
            except Exception as e:
                print(f"error encoding image: {e}")
                return jsonify({'error': 'error encoding image'}), 500
        
        try:
            processed_img_str = image_to_base64(processed_img)
            celeb_img = Image.open(match['img_path'])
            celeb_img_str = image_to_base64(celeb_img)
            
            return jsonify({
                'success': True,
                'match_name': match['name'],
                'similarity': float(similarity),
                'processed_image': processed_img_str,
                'celebrity_image': celeb_img_str,
                'face_count': len(matcher.celeb_data) if matcher else 0
            })
        except Exception as e:
            print(f"error preparing response: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'error preparing response: {str(e)}'}), 500
        
    except Exception as e:
        print(f"error in process_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/register_face', methods=['POST'])
def register_face():
    try:
        data = request.json
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'name required'}), 400
        
        if 'image' not in data:
            return jsonify({'error': 'image required'}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
        filepath = os.path.join(celebs_dir, filename)
        
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            h, w = pil_img.size[1], pil_img.size[0]
            if h > 800 or w > 800:
                scale = min(800.0 / h, 800.0 / w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            pil_img.save(filepath, 'JPEG', quality=95)
            
            rgb = face_recognition.load_image_file(filepath)
            
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            
            rgb = rgb.copy()
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            
            if len(rgb.shape) != 3 or rgb.shape[2] != 3:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'invalid image format'}), 400
            
            is_contiguous = rgb.flags['C_CONTIGUOUS']
            print(f"register_face: rgb shape={rgb.shape}, dtype={rgb.dtype}, min={rgb.min()}, max={rgb.max()}, contiguous={is_contiguous}", flush=True)
            
            face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
            
            if not face_encs or len(face_encs) == 0:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'no face detected in image'}), 400
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"face_encodings error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'face encoding failed: {str(e)}'}), 400
        
        if matcher:
            old_count = len(matcher.celeb_data)
            matcher.load_database()
            new_count = len(matcher.celeb_data)
            print(f"database reloaded after registering {name}: {old_count} -> {new_count} celebs", flush=True)
            
            if os.path.exists(celebs_dir):
                all_files = os.listdir(celebs_dir)
                print(f"all files in celebs dir: {all_files}", flush=True)
                expected_prefix = name.lower().replace(' ', '_')
                matching_files = [f for f in all_files if f.lower().startswith(expected_prefix)]
                print(f"files matching '{expected_prefix}': {matching_files}", flush=True)
            
            found = False
            for celeb in matcher.celeb_data:
                if name.lower().replace(' ', '_') in celeb['name'].lower():
                    print(f"found registered face '{name}' in database as '{celeb['name']}'", flush=True)
                    found = True
                    break
            
            if not found:
                print(f"WARNING: registered face '{name}' not found in database after reload!", flush=True)
                print(f"all celeb names in database: {[c['name'] for c in matcher.celeb_data]}", flush=True)
        
        return jsonify({
            'success': True, 
            'message': f'registered {name} successfully!',
            'count': len(matcher.celeb_data) if matcher else 0
        })
        
    except Exception as e:
        print(f"error in register_face: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def is_face_present(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.6
        )
        results = face_detector.process(rgb_img)
        return results.detections is not None
    except:
        return False


def get_tmdb_image(name):
    try:
        TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
        if not TMDB_API_KEY:
            return None
        url = f"https://api.themoviedb.org/3/search/person"
        params = {"api_key": TMDB_API_KEY, "query": name}
        data = requests.get(url, params=params, timeout=5).json()
        if data.get('results'):
            path = data['results'][0].get('profile_path')
            if path:
                img_url = f"https://image.tmdb.org/t/p/h632{path}"
                resp = requests.get(img_url, timeout=10)
                if resp.status_code == 200:
                    return resp.content
        return None
    except:
        return None


def get_wikipedia_image(name):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "titles": name, "prop": "pageimages",
            "format": "json", "pithumbsize": 1000, "redirects": 1
        }
        resp = requests.get(url, params=params, timeout=5).json()
        pages = resp.get("query", {}).get("pages", {})
        for p in pages.values():
            if "thumbnail" in p:
                img_resp = requests.get(p["thumbnail"]["source"], timeout=10)
                if img_resp.status_code == 200:
                    return img_resp.content
        return None
    except:
        return None


@app.route('/api/suggest_celebrity', methods=['POST'])
def suggest_celebrity():
    try:
        data = request.json
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'celebrity name required'}), 400
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        clean_name = name.replace(" ", "_").lower()
        
        existing_files = [f for f in os.listdir(celebs_dir) if f.lower().startswith(clean_name)]
        if existing_files:
            if matcher:
                matcher.load_database()
            return jsonify({
                'success': True, 
                'message': f'{name} already exists!',
                'already_exists': True
            })
        
        img_data = None
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        img_data = get_tmdb_image(name)
        if img_data and not is_face_present(img_data):
            img_data = None
        
        if not img_data:
            img_data = get_wikipedia_image(name)
            if img_data and not is_face_present(img_data):
                img_data = None
        
        if not img_data:
            for attempt in range(3):
                try:
                    time.sleep(1 + attempt * 2)
                    with DDGS() as ddgs:
                        results = list(ddgs.images(query=f"{name} headshot portrait", max_results=5))
                        for r in results:
                            resp = requests.get(r['image'], timeout=10, headers=headers)
                            if resp.status_code == 200 and is_face_present(resp.content):
                                img_data = resp.content
                                break
                        if img_data:
                            break
                except Exception as e:
                    print(f"ddgs search failed (attempt {attempt+1}): {e}", flush=True)
                    if attempt < 2:
                        time.sleep(2)
        
        if img_data and is_face_present(img_data):
            filename = f"{clean_name}_{int(time.time())}.jpg"
            filepath = os.path.join(celebs_dir, filename)
            
            try:
                pil_img = Image.open(io.BytesIO(img_data))
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                rgb = np.array(pil_img, dtype=np.uint8)
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                h, w = rgb.shape[:2]
                if h > 800 or w > 800:
                    scale = min(800.0 / h, 800.0 / w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    rgb = cv2.resize(rgb, (new_w, new_h))
                    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
                if not face_encs or len(face_encs) == 0:
                    return jsonify({
                        'error': f'no face detected in image for {name}'
                    }), 400
                
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, img_bgr)
            except Exception as e:
                print(f"error validating face for {name}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'failed to process image for {name}: {str(e)}'
                }), 400
            
            if matcher:
                matcher.load_database()
                print(f"database reloaded after adding {name}, total: {len(matcher.celeb_data)}", flush=True)
            
            return jsonify({
                'success': True,
                'message': f'added {name} to database!',
                'count': len(matcher.celeb_data) if matcher else 0
            })
        else:
            return jsonify({
                'error': f'couldnt find a face image for {name}, try different name'
            }), 400
        
    except Exception as e:
        print(f"error in suggest_celebrity: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'no file provided'}), 400
        
        file = request.files['file']
        name = request.form.get('name', '').strip()
        
        if file.filename == '':
            return jsonify({'error': 'no file selected'}), 400
        
        if not name:
            name = os.path.splitext(file.filename)[0]
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
        filepath = os.path.join(celebs_dir, filename)
        
        try:
            pil_img = Image.open(file)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            rgb = np.array(pil_img, dtype=np.uint8)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            
            h, w = rgb.shape[:2]
            if h > 2000 or w > 2000:
                scale = min(2000.0 / h, 2000.0 / w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb = cv2.resize(rgb, (new_w, new_h))
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            
            face_encs = face_recognition.face_encodings(rgb)
            if not face_encs or len(face_encs) == 0:
                return jsonify({'error': 'no face detected in image'}), 400
            
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img_bgr)
        except Exception as e:
            print(f"error processing uploaded image: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'failed to process image: {str(e)}'}), 400
        
        if matcher:
            matcher.load_database()
        
        return jsonify({
            'success': True,
            'message': f'uploaded and registered {name} successfully!',
            'count': len(matcher.celeb_data) if matcher else 0
        })
        
    except Exception as e:
        print(f"error in upload_image: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_bulk', methods=['POST'])
def upload_bulk():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'no files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'no files selected'}), 400
        
        celebs_dir = "celebs"
        if not os.path.exists(celebs_dir):
            os.makedirs(celebs_dir, exist_ok=True)
        
        added = 0
        failed = 0
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            name = os.path.splitext(file.filename)[0]
            filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
            filepath = os.path.join(celebs_dir, filename)
            
            try:
                pil_img = Image.open(file)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                h, w = pil_img.size[1], pil_img.size[0]
                if h > 800 or w > 800:
                    scale = min(800.0 / h, 800.0 / w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                pil_img.save(filepath, 'JPEG', quality=95)
                
                rgb = face_recognition.load_image_file(filepath)
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                
                face_encs = face_recognition.face_encodings(rgb, num_jitters=1)
                if not face_encs or len(face_encs) == 0:
                    os.remove(filepath)
                    failed += 1
                    results.append({'file': file.filename, 'status': 'failed', 'reason': 'no face detected'})
                    continue
                
                added += 1
                results.append({'file': file.filename, 'status': 'success', 'name': name})
                
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                failed += 1
                print(f"error processing {file.filename}: {e}", flush=True)
                results.append({'file': file.filename, 'status': 'failed', 'reason': str(e)})
        
        if matcher and added > 0:
            matcher.load_database()
        
        return jsonify({
            'success': True,
            'message': f'uploaded {added} images successfully, {failed} failed',
            'added': added,
            'failed': failed,
            'results': results,
            'total_count': len(matcher.celeb_data) if matcher else 0
        })
        
    except Exception as e:
        print(f"error in upload_bulk: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production' and not os.environ.get('RAILWAY_ENVIRONMENT')
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
