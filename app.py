from flask import Flask, request, render_template, send_from_directory, url_for
from moviepy.editor import VideoFileClip, vfx, concatenate_videoclips, CompositeVideoClip, AudioFileClip
from moviepy.editor import TextClip, ImageClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import math
import time
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
STORYBOARD_OUTPUT_FOLDER = 'storyboard_output'
AUDIO_FOLDER = 'audio'
IMAGES_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STORYBOARD_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files.get('video')
        form_data = request.form.to_dict()
        
        if not video:
            return render_template('index.html', form_data=form_data)

        action = form_data.get('action')
        filename = video.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(input_path)
        # If not mp4, convert to mp4
        base, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext != '.mp4':
            mp4_filename = base + '.mp4'
            mp4_path = os.path.join(UPLOAD_FOLDER, mp4_filename)
            # Use ffmpeg to convert
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-strict', 'experimental', mp4_path
                ], check=True)
                filename = mp4_filename
                input_path = mp4_path
            except Exception as e:
                return render_template('index.html', form_data=form_data, error=f'Error converting video to mp4: {e}')


        start_hours = int(form_data.get('start_hours', 0))
        start_minutes = int(form_data.get('start_minutes', 0))
        start_seconds = int(form_data.get('start_seconds', 0))
        end_hours_str = form_data.get('end_hours')
        end_minutes_str = form_data.get('end_minutes')
        end_seconds_str = form_data.get('end_seconds')
        start_time = start_hours * 3600 + start_minutes * 60 + start_seconds
        end_hours = int(end_hours_str) if end_hours_str and end_hours_str.isdigit() else None
        end_minutes = int(end_minutes_str) if end_minutes_str and end_minutes_str.isdigit() else None
        end_seconds = int(end_seconds_str) if end_seconds_str and end_seconds_str.isdigit() else None
        if end_hours is None or end_minutes is None or end_seconds is None:
            end_time = None
        else:
            end_time = end_hours * 3600 + end_minutes * 60 + end_seconds
            


        if action == 'split':
            aspect_ratio = form_data.get('aspect_ratio', 'original')
            resolution = form_data.get('resolution', '1080')
            chunk_length = form_data.get('chunk_length', '30')
            if chunk_length == 'custom':
                try:
                    chunk_length = int(form_data.get('custom_chunk_length', 30))
                except Exception:
                    chunk_length = 30
            else:
                chunk_length = int(chunk_length)
            color_grade = form_data.get('color_grade', 'none')
            
            clips = split_video(input_path, filename, start_time=start_time, end_time=end_time,
                                  aspect_ratio=aspect_ratio, resolution=resolution, chunk_length=chunk_length, color_grade=color_grade)
            # Auto-Thumbnail Generator: extract a thumbnail from the first output clip
            if clips:
                thumb_name = extract_thumbnail(os.path.join(OUTPUT_FOLDER, clips[0]), OUTPUT_FOLDER, clips[0])
            else:
                thumb_name = None
            return render_template('index.html', clips=clips, form_data=form_data, thumbnail=thumb_name)

        elif action == 'storyboard':
            aspect_ratio = form_data.get('aspect_ratio', 'original')
            resolution = form_data.get('resolution', '1080')
            storyboard_chunk_length = form_data.get('storyboard_chunk_length', 10)
            if isinstance(storyboard_chunk_length, str) and storyboard_chunk_length.isdigit():
                storyboard_chunk_length = int(storyboard_chunk_length)
            storyboard_clips = create_storyboard(input_path, filename, start_time=start_time, end_time=end_time,
                                               chunk_length=storyboard_chunk_length, aspect_ratio=aspect_ratio,
                                               resolution=resolution)
            # Auto-Thumbnail Generator: extract a thumbnail from the first storyboard clip
            if storyboard_clips:
                thumb_name = extract_thumbnail(os.path.join(STORYBOARD_OUTPUT_FOLDER, storyboard_clips[0]), STORYBOARD_OUTPUT_FOLDER, storyboard_clips[0])
            else:
                thumb_name = None
            return render_template('index.html', storyboard_clips=storyboard_clips, storyboard_folder=True, form_data=form_data, thumbnail=thumb_name)

        elif action == 'change_audio':
            audio = request.files.get('audio')
            if not audio:
                return render_template('index.html', form_data=form_data, error="No audio file provided.")

            audio_filename = audio.filename
            audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
            audio.save(audio_path)

            new_clip_name = change_audio(input_path, filename, audio_path)
            # Auto-Thumbnail Generator: extract a thumbnail from the new audio clip
            thumb_name = extract_thumbnail(os.path.join(OUTPUT_FOLDER, new_clip_name), OUTPUT_FOLDER, new_clip_name)
            return render_template('index.html', clips=[new_clip_name], form_data=form_data, audio_changed=True, thumbnail=thumb_name)

        elif action == 'change_speed':
            speed_factor = float(form_data.get('speed_factor', 1.0))
            new_clip_name = change_video_speed(input_path, filename, speed_factor)
            # Auto-Thumbnail Generator: extract a thumbnail from the speed-changed clip
            thumb_name = extract_thumbnail(os.path.join(OUTPUT_FOLDER, new_clip_name), OUTPUT_FOLDER, new_clip_name)
            return render_template('index.html', clips=[new_clip_name], form_data=form_data, speed_changed=True, thumbnail=thumb_name)

        elif action == 'merge':
            return render_template('index.html', form_data=form_data)

        elif action == 'create_collage':
            images = request.files.getlist('collage_images')
            layout = form_data.get('collage_layout', 'grid_2x2')
            aspect_ratio = form_data.get('collage_aspect_ratio', '9:16')
            background_color = form_data.get('background_color', '#ffffff')
            
            if not images or len(images) < 2:
                return render_template('index.html', form_data=form_data, collage_error="Please upload at least 2 images for collage.")
            
            collage_filename = create_simple_collage(images, layout, aspect_ratio, background_color)
            return render_template('index.html', form_data=form_data, collage_result=collage_filename)
        
        elif action == 'editor_upload':
            # Handle image upload for editor
            images = request.files.getlist('editor_images')
            uploaded_images = []
            
            for image in images:
                if image.filename:
                    filename = f"{int(time.time())}_{image.filename}"
                    image_path = os.path.join(IMAGES_FOLDER, filename)
                    image.save(image_path)
                    uploaded_images.append({'filename': filename})
            
            return render_template('index.html', form_data=form_data, editor_images=uploaded_images)
        
        elif action == 'save_canvas':
            # Handle canvas save from editor
            canvas_data = form_data.get('canvas_data')
            if canvas_data:
                # Remove data URL prefix
                canvas_data = canvas_data.replace('data:image/png;base64,', '')
                
                # Save the image
                import base64
                image_data = base64.b64decode(canvas_data)
                filename = f"custom_collage_{int(time.time())}.png"
                output_path = os.path.join(OUTPUT_FOLDER, filename)
                
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                
                return render_template('index.html', form_data=form_data, canvas_result=filename)

    return render_template('index.html', form_data={})

@app.route('/merge', methods=['POST'])
def merge_clips():
    files = request.files.getlist('clips')
    clip_order = request.form.get('clip_order')
    if not files or len(files) < 2:
        return render_template('index.html', merge_error='Please upload at least two clips to merge.', form_data={})
    # Save files and map filename to file object
    file_map = {}
    for file in files:
        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        file_map[filename] = path
    # Determine order
    ordered_paths = []
    if clip_order:
        order = [name for name in clip_order.split(',') if name in file_map]
        ordered_paths = [file_map[name] for name in order]
    else:
        ordered_paths = list(file_map.values())
    try:
        video_clips = [VideoFileClip(path) for path in ordered_paths]
        # Use method='chain' for gapless, fluid merging
        final_clip = concatenate_videoclips(video_clips, method='chain')
        outname = 'merged_clip.mp4'
        outpath = os.path.join(OUTPUT_FOLDER, outname)
        final_clip.write_videofile(outpath, codec='libx264', audio_codec='aac')
        for clip in video_clips:
            clip.close()
        final_clip.close()
        return render_template('index.html', merged_clip=outname, form_data={})
    except Exception as e:
        return render_template('index.html', merge_error=f'Error merging clips: {str(e)}', form_data={})

# Helper function to find the center of motion in a video segment

def find_motion_center(video_path, start_time=0, end_time=None, sample_rate=5):
    """
    Analyzes the video between start_time and end_time and returns the (x, y) center
    of the region with the most motion, using frame differencing.
    sample_rate: sample every N frames for speed.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_time is None:
        end_time = total_frames / fps
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_gray = None
    motion_map = np.zeros((height, width), dtype=np.float32)
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % sample_rate != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_map += diff.astype(np.float32)
        prev_gray = gray
        frame_idx += 1
    cap.release()
    # Find the centroid of the largest motion area
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(motion_map)
    # max_loc is (x, y) of the most motion
    return max_loc

# Helper function to find the center of focus (face or motion) in a video segment

def find_focus_center(video_path, start_time=0, end_time=None, sample_rate=5):
    """
    Tries to find the average face center in the segment. If no face is found, falls back to motion center.
    Returns (x, y) center.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_time is None:
        end_time = total_frames / fps
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_centers = []
    prev_gray = None
    motion_map = np.zeros((height, width), dtype=np.float32)
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % sample_rate != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            face_centers.append((x + w // 2, y + h // 2))
        # For fallback: motion map
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_map += diff.astype(np.float32)
        prev_gray = gray
        frame_idx += 1
    cap.release()
    if face_centers:
        # Average face center
        arr = np.array(face_centers)
        avg_x = int(np.mean(arr[:, 0]))
        avg_y = int(np.mean(arr[:, 1]))
        return (avg_x, avg_y)
    # Fallback: motion center
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(motion_map)
    return max_loc

def apply_color_grade(clip, grade):
    import cv2
    import numpy as np
    if grade == 'none':
        return clip
    def teal_orange(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.1, 0, 255)
        lut_g = np.clip(lut * 0.95, 0, 255)
        lut_b = np.clip(lut * 1.2 - 30, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def blockbuster(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_b = np.clip(lut * 1.15, 0, 255)
        lut_g = np.clip(lut * 0.95, 0, 255)
        lut_r = np.clip(lut * 0.9, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def warm_film(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.1 + 10, 0, 255)
        lut_g = np.clip(lut * 1.05, 0, 255)
        lut_b = np.clip(lut * 0.9, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def bw(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    def vintage(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 0.9 + 20, 0, 255)
        lut_g = np.clip(lut * 0.85 + 15, 0, 255)
        lut_b = np.clip(lut * 0.8 + 10, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def cyberpunk(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.3 + 30, 0, 255)
        lut_g = np.clip(lut * 0.7, 0, 255)
        lut_b = np.clip(lut * 1.4 + 40, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def noir(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Add contrast
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=-20)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def sunset(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.4 + 40, 0, 255)
        lut_g = np.clip(lut * 1.2 + 20, 0, 255)
        lut_b = np.clip(lut * 0.8, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def cool_blue(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 0.8, 0, 255)
        lut_g = np.clip(lut * 0.9, 0, 255)
        lut_b = np.clip(lut * 1.3 + 30, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def warm_gold(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.3 + 30, 0, 255)
        lut_g = np.clip(lut * 1.2 + 20, 0, 255)
        lut_b = np.clip(lut * 0.7, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def high_contrast(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=-30)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def sepia(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Apply sepia filter
        img = img.astype(np.float32)
        img = img * np.array([0.393, 0.769, 0.189])
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    def neon(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.5 + 50, 0, 255)
        lut_g = np.clip(lut * 1.4 + 40, 0, 255)
        lut_b = np.clip(lut * 1.6 + 60, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def cinematic(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.1 + 15, 0, 255)
        lut_g = np.clip(lut * 0.95 + 5, 0, 255)
        lut_b = np.clip(lut * 1.2 + 25, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def dramatic(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.convertScaleAbs(img, alpha=1.4, beta=-40)
        # Add slight color tint
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 1.1 + 10, 0, 255)
        lut_g = np.clip(lut * 0.9, 0, 255)
        lut_b = np.clip(lut * 1.15 + 15, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def pastel(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_r = np.clip(lut * 0.8 + 50, 0, 255)
        lut_g = np.clip(lut * 0.8 + 50, 0, 255)
        lut_b = np.clip(lut * 0.8 + 50, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img[..., 1] = cv2.LUT(img[..., 1], lut_g)
        img[..., 2] = cv2.LUT(img[..., 2], lut_r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def monochrome(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Add slight blue tint
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lut = np.arange(256, dtype=np.uint8)
        lut_b = np.clip(lut * 1.1 + 10, 0, 255)
        img[..., 0] = cv2.LUT(img[..., 0], lut_b)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    fn_map = {
        'teal_orange': teal_orange,
        'blockbuster': blockbuster,
        'warm_film': warm_film,
        'bw': bw,
        'vintage': vintage,
        'cyberpunk': cyberpunk,
        'noir': noir,
        'sunset': sunset,
        'cool_blue': cool_blue,
        'warm_gold': warm_gold,
        'high_contrast': high_contrast,
        'sepia': sepia,
        'neon': neon,
        'cinematic': cinematic,
        'dramatic': dramatic,
        'pastel': pastel,
        'monochrome': monochrome
    }
    fn = fn_map.get(grade, None)
    if fn:
        return clip.fl_image(fn)
    return clip

# Update split_video to accept color_grade and apply it

def split_video(path, filename, start_time=0, end_time=None, aspect_ratio='original', resolution='1080', chunk_length=30, color_grade='none'):
    try:
        if not os.path.exists(path):
            print(f"Error: Input file '{path}' not found.")
            return []
        clip = VideoFileClip(path)
        start_time = max(0, start_time)
        if end_time is None or end_time > clip.duration:
            end_time = clip.duration
        if start_time >= end_time:
            print(f"Error: Start time ({start_time}s) must be before end time ({end_time}s).")
            return []
        main_clip = clip.subclip(start_time, end_time)
        if aspect_ratio == '9:16':
            w, h = main_clip.size
            target_ratio = 9.0 / 16.0
            motion_x, motion_y = find_motion_center(path, start_time, end_time)
            if (w / h) > target_ratio:
                new_w = int(h * target_ratio)
                x_center = np.clip(motion_x, new_w // 2, w - new_w // 2)
                main_clip = main_clip.fx(vfx.crop, x_center=x_center, width=new_w)
            else:
                new_h = int(w / target_ratio)
                y_center = np.clip(motion_y, new_h // 2, h - new_h // 2)
                main_clip = main_clip.fx(vfx.crop, y_center=y_center, height=new_h)
        res_map = {'720': 720, '1080': 1080, '2160': 2160}
        if resolution != 'original':
            target_width = res_map.get(resolution, 1080)
            if aspect_ratio == '9:16':
                main_clip = main_clip.fx(vfx.resize, width=target_width)
            elif main_clip.size[0] > target_width:
                main_clip = main_clip.fx(vfx.resize, width=target_width)
        duration = math.ceil(main_clip.duration)
        clip_list = []
        chunk_length = max(1, chunk_length)
        base_filename = os.path.splitext(filename)[0]
        for i in range(0, duration, chunk_length):
            try:
                subclip = main_clip.subclip(i, min(i + chunk_length, duration))
                subclip = apply_color_grade(subclip, color_grade)
                outname = f"{base_filename}_part{i//chunk_length + 1}.mp4"
                outpath = os.path.join(OUTPUT_FOLDER, outname)
                subclip.write_videofile(outpath, codec="libx264", audio_codec="aac", logger=None)
                clip_list.append(outname)
            except Exception as e:
                print(f"Error processing clip segment {i//chunk_length + 1}: {str(e)}")
                continue
        main_clip.close()
        clip.close()
        return clip_list
    except Exception as e:
        print(f"Error in split_video: {str(e)}")
        return []

def create_storyboard(path, filename, start_time=0, end_time=None, chunk_length=10, aspect_ratio='original', resolution='1080', fade_duration=0.5):
    clip = VideoFileClip(path)
    if end_time is None or end_time > clip.duration:
        end_time = clip.duration
    main_clip = clip.subclip(start_time, end_time)
    if aspect_ratio == '9:16':
        w, h = main_clip.size
        target_ratio = 9.0 / 16.0
        motion_x, motion_y = find_motion_center(path, start_time, end_time)
        if (w / h) > target_ratio:
            new_w = int(h * target_ratio)
            x_center = np.clip(motion_x, new_w // 2, w - new_w // 2)
            main_clip = main_clip.fx(vfx.crop, x_center=x_center, width=new_w)
        else:
            new_h = int(w / target_ratio)
            y_center = np.clip(motion_y, new_h // 2, h - new_h // 2)
            main_clip = main_clip.fx(vfx.crop, y_center=y_center, height=new_h)
    res_map = {'720': 720, '1080': 1080, '2160': 2160}
    if resolution != 'original':
        target_width = res_map.get(resolution, 1080)
        if aspect_ratio == '9:16':
            main_clip = main_clip.fx(vfx.resize, width=target_width)
        elif main_clip.size[0] > target_width:
            main_clip = main_clip.fx(vfx.resize, width=target_width)
    duration = math.ceil(main_clip.duration)
    clip_list = []
    base_filename = os.path.splitext(filename)[0]
    for i in range(0, duration, chunk_length):
        start = i
        end = min(i + chunk_length, duration)
        if start >= end:
            continue
        subclip = main_clip.subclip(start, end)
        subclip = subclip.fx(vfx.fadein, duration=fade_duration)
        subclip = subclip.fx(vfx.fadeout, duration=fade_duration)
        outname = f"{base_filename}_storyboard_part{i//chunk_length + 1}.mp4"
        outpath = os.path.join(STORYBOARD_OUTPUT_FOLDER, outname)
        subclip.write_videofile(outpath, codec="libx264", audio_codec="aac")
        clip_list.append(outname)
        subclip.close()
    main_clip.close()
    clip.close()
    return clip_list

def change_audio(video_path, video_filename, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # If the new audio is shorter than the video, loop it
    if audio_clip.duration < video_clip.duration:
        audio_clip = audio_clip.fx(vfx.loop, duration=video_clip.duration)
    else:
        # If the new audio is longer, trim it to the video's duration
        audio_clip = audio_clip.subclip(0, video_clip.duration)

    new_video = video_clip.set_audio(audio_clip)
    
    
    base_filename = os.path.splitext(video_filename)[0]
    outname = f"{base_filename}_new_audio.mp4"
    outpath = os.path.join(OUTPUT_FOLDER, outname)
    
    new_video.write_videofile(outpath, codec="libx264", audio_codec="aac", temp_audiofile=os.path.join(os.path.dirname(outpath), "temp-audio.m4a"))
    new_video.close()
    video_clip.close()
    audio_clip.close()
    return outname


def change_video_speed(video_path, video_filename, speed_factor):
    """
    Change the speed of a video clip.
    speed_factor: float - 0.5 for 2x slower, 2.0 for 2x faster, etc.
    """
    try:
        video_clip = VideoFileClip(video_path)
        
        # Apply speed change using MoviePy's speedx effect
        speeded_clip = video_clip.fx(vfx.speedx, speed_factor)
        
        base_filename = os.path.splitext(video_filename)[0]
        speed_label = f"{speed_factor}x" if speed_factor != 1.0 else "normal"
        outname = f"{base_filename}_speed_{speed_label}.mp4"
        outpath = os.path.join(OUTPUT_FOLDER, outname)
        
        # Write the speeded video
        speeded_clip.write_videofile(outpath, codec="libx264", audio_codec="aac", logger=None)
        
        # Clean up
        speeded_clip.close()
        video_clip.close()
        
        return outname
        
    except Exception as e:
        print(f"Error changing video speed: {e}")
        raise e


def extract_thumbnail(video_path, output_folder, filename, time_ratio=0.33):
    """
    Extract a frame at a given ratio of the video duration and save as a thumbnail image.
    Returns the thumbnail filename.
    """
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        t = duration * time_ratio
        frame = clip.get_frame(t)
        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame)
        base_filename = os.path.splitext(filename)[0]
        thumb_name = f"{base_filename}_thumbnail.jpg"
        thumb_path = os.path.join(output_folder, thumb_name)
        img.save(thumb_path)
        clip.close()
        return thumb_name
    except Exception as e:
        print(f"Error extracting thumbnail: {e}")
        return None

@app.route('/output/<filename>')
def download_clip(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)

@app.route('/storyboard_output/<filename>')
def download_storyboard_clip(filename):
    return send_from_directory(STORYBOARD_OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/thumbnail/<filename>')
def serve_thumbnail(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/download_thumbnail/<filename>')
def download_thumbnail(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/separate', methods=['POST'])
def separate_audio_video():
    try:
        video = request.files.get('video')
        audio = request.files.get('audio')
        video_format = request.form.get('video_format', 'mp4')
        audio_format = request.form.get('audio_format', 'mp3')
        enable_audio_split = request.form.get('enable_audio_split') == 'on'
        audio_start_time = request.form.get('audio_start_time', '00:00.00')
        audio_end_time = request.form.get('audio_end_time', '00:00.00')
        
        if not video and not audio:
            return render_template('index.html', form_data={}, separation_error='No file uploaded')
        
        if video:
            # Process video file
            filename = video.filename
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            video.save(input_path)
            
            # Separate audio and video
            separated_files = separate_audio_from_video(
                input_path, filename, video_format, audio_format,
                enable_audio_split, audio_start_time, audio_end_time
            )
        else:
            # Process audio file
            filename = audio.filename
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            audio.save(input_path)
            
            # Create audio clip from audio file
            separated_files = create_audio_clip_from_audio(
                input_path, filename, audio_format,
                enable_audio_split, audio_start_time, audio_end_time
            )
        
        return render_template('index.html', form_data={}, separated_files=separated_files)
        
    except Exception as e:
        return render_template('index.html', form_data={}, separation_error=f'Error processing file: {str(e)}')

@app.route('/separated/<filename>')
def download_separated(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

def separate_audio_from_video(video_path, filename, video_format='mp4', audio_format='mp3', 
                             enable_audio_split=False, audio_start_time='00:00.00', audio_end_time='00:00.00'):
    """
    Separate audio and video from a video file and save them as separate files.
    Optionally extract a specific audio clip based on time range.
    Returns a dictionary with 'video', 'audio', and optionally 'audio_clip' filenames.
    """
    try:
        # Load the video clip
        clip = VideoFileClip(video_path)
        base_name = os.path.splitext(filename)[0]
        
        # Generate output filenames
        video_filename = f"{base_name}_video_only.{video_format}"
        audio_filename = f"{base_name}_audio_only.{audio_format}"
        
        video_output_path = os.path.join(OUTPUT_FOLDER, video_filename)
        audio_output_path = os.path.join(OUTPUT_FOLDER, audio_filename)
        
        # Extract audio without video
        audio_clip = clip.audio
        audio_clip_filename = None
        
        if audio_clip is not None:
            # Determine audio codec based on format
            audio_codec_map = {
                'mp3': 'mp3',
                'wav': 'pcm_s16le',
                'aac': 'aac',
                'flac': 'flac'
            }
            audio_codec = audio_codec_map.get(audio_format, 'mp3')
            
            # Write full audio file
            audio_clip.write_audiofile(audio_output_path, codec=audio_codec)
            
            # If audio split is enabled, create a clip
            if enable_audio_split:
                try:
                    # Parse time strings (format: MM:SS.ms)
                    start_parts = audio_start_time.split(':')
                    end_parts = audio_end_time.split(':')
                    
                    start_minutes = int(start_parts[0])
                    start_seconds = float(start_parts[1])
                    start_time_seconds = start_minutes * 60 + start_seconds
                    
                    end_minutes = int(end_parts[0])
                    end_seconds = float(end_parts[1])
                    end_time_seconds = end_minutes * 60 + end_seconds
                    
                    # Ensure times are within bounds
                    start_time_seconds = max(0, min(start_time_seconds, clip.duration))
                    end_time_seconds = max(start_time_seconds + 0.1, min(end_time_seconds, clip.duration))
                    
                    # Create audio clip
                    audio_clip_segment = audio_clip.subclip(start_time_seconds, end_time_seconds)
                    audio_clip_filename = f"{base_name}_audio_clip_{start_time_seconds:.1f}s_to_{end_time_seconds:.1f}s.{audio_format}"
                    audio_clip_path = os.path.join(OUTPUT_FOLDER, audio_clip_filename)
                    
                    audio_clip_segment.write_audiofile(audio_clip_path, codec=audio_codec)
                    audio_clip_segment.close()
                    
                except Exception as clip_error:
                    print(f"Error creating audio clip: {clip_error}")
                    audio_clip_filename = None
            
            audio_clip.close()
        else:
            audio_filename = None
        
        # Extract video without audio
        video_clip = clip.without_audio()
        
        # Determine video codec based on format
        video_codec_map = {
            'mp4': 'libx264',
            'avi': 'libx264',
            'mov': 'libx264',
            'mkv': 'libx264'
        }
        video_codec = video_codec_map.get(video_format, 'libx264')
        
        # Write video without audio
        video_clip.write_videofile(video_output_path, codec=video_codec, audio=False)
        video_clip.close()
        clip.close()
        
        result = {
            'video': video_filename,
            'audio': audio_filename
        }
        
        if audio_clip_filename:
            result['audio_clip'] = audio_clip_filename
        
        return result
        
    except Exception as e:
        print(f"Error separating audio and video: {e}")
        raise e

def create_audio_clip_from_audio(audio_path, filename, audio_format='mp3', 
                                enable_audio_split=False, audio_start_time='00:00.00', audio_end_time='00:00.00'):
    """
    Create audio clips from an audio file.
    Returns a dictionary with 'audio' and optionally 'audio_clip' filenames.
    """
    try:
        # Load the audio clip
        audio_clip = AudioFileClip(audio_path)
        base_name = os.path.splitext(filename)[0]
        
        # Generate output filename
        audio_filename = f"{base_name}_processed.{audio_format}"
        audio_output_path = os.path.join(OUTPUT_FOLDER, audio_filename)
        
        # Determine audio codec based on format
        audio_codec_map = {
            'mp3': 'mp3',
            'wav': 'pcm_s16le',
            'aac': 'aac',
            'flac': 'flac'
        }
        audio_codec = audio_codec_map.get(audio_format, 'mp3')
        
        # Write full audio file
        audio_clip.write_audiofile(audio_output_path, codec=audio_codec)
        
        audio_clip_filename = None
        
        # If audio split is enabled, create a clip
        if enable_audio_split:
            try:
                # Parse time strings (format: MM:SS.ms)
                start_parts = audio_start_time.split(':')
                end_parts = audio_end_time.split(':')
                
                start_minutes = int(start_parts[0])
                start_seconds = float(start_parts[1])
                start_time_seconds = start_minutes * 60 + start_seconds
                
                end_minutes = int(end_parts[0])
                end_seconds = float(end_parts[1])
                end_time_seconds = end_minutes * 60 + end_seconds
                
                # Ensure times are within bounds
                start_time_seconds = max(0, min(start_time_seconds, audio_clip.duration))
                end_time_seconds = max(start_time_seconds + 0.1, min(end_time_seconds, audio_clip.duration))
                
                # Create audio clip
                audio_clip_segment = audio_clip.subclip(start_time_seconds, end_time_seconds)
                audio_clip_filename = f"{base_name}_audio_clip_{start_time_seconds:.1f}s_to_{end_time_seconds:.1f}s.{audio_format}"
                audio_clip_path = os.path.join(OUTPUT_FOLDER, audio_clip_filename)
                
                audio_clip_segment.write_audiofile(audio_clip_path, codec=audio_codec)
                audio_clip_segment.close()
                
            except Exception as clip_error:
                print(f"Error creating audio clip: {clip_error}")
                audio_clip_filename = None
        
        audio_clip.close()
        
        result = {
            'audio': audio_filename
        }
        
        if audio_clip_filename:
            result['audio_clip'] = audio_clip_filename
        
        return result
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise e


def create_image_collage(images, layout='grid_2x2', aspect_ratio='1:1', background_color='#ffffff', spacing=10):
    """
    Create an image collage from multiple uploaded images.
    
    Args:
        images: List of uploaded image files
        layout: Layout type (grid_2x2, grid_3x3, horizontal, vertical, triptych, etc.)
        aspect_ratio: Final collage aspect ratio
        background_color: Background color for the collage
        spacing: Spacing between images in pixels
    
    Returns:
        str: Filename of the created collage
    """
    try:
        # Save uploaded images and load them
        image_list = []
        for i, image_file in enumerate(images):
            if image_file and image_file.filename:
                # Save the uploaded image
                image_path = os.path.join(IMAGES_FOLDER, f"temp_{i}_{image_file.filename}")
                image_file.save(image_path)
                
                # Load and process the image
                img = Image.open(image_path)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_list.append(img)
        
        if not image_list:
            raise ValueError("No valid images provided")
        
        # Determine collage dimensions based on aspect ratio
        if aspect_ratio == '1:1':
            width, height = 1080, 1080  # Square
        elif aspect_ratio == '4:5':
            width, height = 1080, 1350  # Instagram portrait
        elif aspect_ratio == '16:9':
            width, height = 1920, 1080  # Landscape
        elif aspect_ratio == '9:16':
            width, height = 1080, 1920  # Portrait
        else:
            width, height = 1080, 1080  # Default square
        
        # Create background
        background = Image.new('RGB', (width, height), background_color)
        
        # Apply layout based on number of images and layout type
        if layout == 'grid_2x2':
            collage = create_grid_layout(image_list, 2, 2, width, height, spacing)
        elif layout == 'horizontal':
            collage = create_horizontal_layout(image_list, width, height, spacing)
        elif layout == 'vertical':
            collage = create_vertical_layout(image_list, width, height, spacing)
        elif layout == 'triptych':
            collage = create_triptych_layout(image_list, width, height, spacing)
        else:
            # Default to grid layout
            collage = create_grid_layout(image_list, 2, 2, width, height, spacing)
        
        # Save the collage
        timestamp = int(time.time())
        collage_filename = f"collage_{timestamp}.jpg"
        collage_path = os.path.join(OUTPUT_FOLDER, collage_filename)
        collage.save(collage_path, 'JPEG', quality=95)
        
        # Clean up temporary files
        for i in range(len(images)):
            temp_path = os.path.join(IMAGES_FOLDER, f"temp_{i}_{images[i].filename}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return collage_filename
        
    except Exception as e:
        print(f"Error creating image collage: {e}")
        raise e


def create_grid_layout(images, rows, cols, width, height, background_color='#ffffff'):
    """Create a grid layout collage."""
    background = Image.new('RGB', (width, height), background_color)
    
    # Calculate cell dimensions
    cell_width = width // cols
    cell_height = height // rows
    
    for i, img in enumerate(images[:rows*cols]):
        row = i // cols
        col = i % cols
        
        # Resize image to fill cell completely
        resized_img = resize_image_to_fill(img, cell_width, cell_height, background_color)
        
        # Calculate position
        x = col * cell_width
        y = row * cell_height
        
        # Paste image
        background.paste(resized_img, (x, y))
    
    return background


def create_horizontal_layout(images, width, height, background_color='#ffffff'):
    """Create a horizontal layout collage for Instagram Reels."""
    background = Image.new('RGB', (width, height), background_color)
    
    # Calculate image dimensions
    num_images = len(images)
    if num_images == 0:
        return background
    
    # For Instagram Reels, we want images to take full height and divide width
    image_width = width // num_images
    image_height = height
    
    for i, img in enumerate(images):
        # Resize image to fill the cell completely
        resized_img = resize_image_to_fill(img, image_width, image_height, background_color)
        
        # Calculate position
        x = i * image_width
        y = 0
        
        # Paste image
        background.paste(resized_img, (x, y))
    
    return background


def create_vertical_layout(images, width, height, background_color='#ffffff'):
    """Create a vertical layout collage for Instagram Reels."""
    background = Image.new('RGB', (width, height), background_color)
    
    # Calculate image dimensions
    num_images = len(images)
    if num_images == 0:
        return background
    
    # For Instagram Reels, we want images to take full width and divide height
    image_width = width
    image_height = height // num_images
    
    for i, img in enumerate(images):
        # Resize image to fill the cell completely
        resized_img = resize_image_to_fill(img, image_width, image_height, background_color)
        
        # Calculate position
        x = 0
        y = i * image_height
        
        # Paste image
        background.paste(resized_img, (x, y))
    
    return background


def create_triptych_layout(images, width, height, background_color='#ffffff'):
    """Create a triptych layout for Instagram Reels (three equal parts)."""
    background = Image.new('RGB', (width, height), background_color)
    
    if len(images) < 3:
        # If less than 3 images, use horizontal layout
        return create_horizontal_layout(images, width, height, background_color)
    
    # Calculate dimensions - three equal parts
    part_width = width // 3
    image_height = height
    
    # Left image
    if len(images) > 0:
        left_img = resize_image_to_fill(images[0], part_width, image_height, background_color)
        x = 0
        y = 0
        background.paste(left_img, (x, y))
    
    # Center image
    if len(images) > 1:
        center_img = resize_image_to_fill(images[1], part_width, image_height, background_color)
        x = part_width
        y = 0
        background.paste(center_img, (x, y))
    
    # Right image
    if len(images) > 2:
        right_img = resize_image_to_fill(images[2], part_width, image_height, background_color)
        x = 2 * part_width
        y = 0
        background.paste(right_img, (x, y))
    
    return background





def resize_image_to_fit(image, target_width, target_height):
    """Resize image to fit within target dimensions while maintaining aspect ratio."""
    # Calculate aspect ratios
    img_ratio = image.width / image.height
    target_ratio = target_width / target_height
    
    if img_ratio > target_ratio:
        # Image is wider than target, fit to width
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        # Image is taller than target, fit to height
        new_height = target_height
        new_width = int(target_height * img_ratio)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new image with target dimensions and white background
    result = Image.new('RGB', (target_width, target_height), '#ffffff')
    
    # Calculate centering position
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    
    # Paste resized image
    result.paste(resized, (x, y))
    
    return result

def resize_image_to_fill(image, target_width, target_height, background_color='#ffffff'):
    """Resize image to fill target dimensions completely (crop if necessary)."""
    # Calculate aspect ratios
    img_ratio = image.width / image.height
    target_ratio = target_width / target_height
    
    if img_ratio > target_ratio:
        # Image is wider than target, fit to height and crop width
        new_height = target_height
        new_width = int(target_height * img_ratio)
    else:
        # Image is taller than target, fit to width and crop height
        new_width = target_width
        new_height = int(target_width / img_ratio)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new image with target dimensions and specified background
    result = Image.new('RGB', (target_width, target_height), background_color)
    
    # Calculate centering position for cropping
    x = (new_width - target_width) // 2
    y = (new_height - target_height) // 2
    
    # Crop and paste the center portion
    if img_ratio > target_ratio:
        # Crop from center horizontally
        cropped = resized.crop((x, 0, x + target_width, target_height))
    else:
        # Crop from center vertically
        cropped = resized.crop((0, y, target_width, y + target_height))
    
    result.paste(cropped, (0, 0))
    
    return result



def create_simple_collage(images, layout='grid_2x2', aspect_ratio='9:16', background_color='#ffffff'):
    """Create a simple image collage."""
    try:
        # Save uploaded images and load them
        image_list = []
        for i, image_file in enumerate(images):
            if image_file and image_file.filename:
                # Save the uploaded image
                image_path = os.path.join(IMAGES_FOLDER, f"temp_{i}_{image_file.filename}")
                image_file.save(image_path)
                
                # Load and process the image
                img = Image.open(image_path)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_list.append(img)
        
        if not image_list:
            raise ValueError("No valid images provided")
        
        # Determine collage dimensions based on aspect ratio
        if aspect_ratio == '1:1':
            width, height = 1080, 1080  # Square
        elif aspect_ratio == '4:5':
            width, height = 1080, 1350  # Instagram portrait
        elif aspect_ratio == '16:9':
            width, height = 1920, 1080  # Landscape
        elif aspect_ratio == '9:16':
            width, height = 1080, 1920  # Instagram Reels
        else:
            width, height = 1080, 1920  # Default to Instagram Reels
        
        # Create background
        background = Image.new('RGB', (width, height), background_color)
        
        # Apply layout
        if layout == 'grid_2x2':
            collage = create_grid_layout(image_list, 2, 2, width, height, background_color)
        elif layout == 'horizontal':
            collage = create_horizontal_layout(image_list, width, height, background_color)
        elif layout == 'vertical':
            collage = create_vertical_layout(image_list, width, height, background_color)
        elif layout == 'triptych':
            collage = create_triptych_layout(image_list, width, height, background_color)
        else:
            collage = create_grid_layout(image_list, 2, 2, width, height, background_color)
        
        # Save the collage
        timestamp = int(time.time())
        collage_filename = f"collage_{timestamp}.jpg"
        collage_path = os.path.join(OUTPUT_FOLDER, collage_filename)
        collage.save(collage_path, 'JPEG', quality=95)
        
        # Clean up temporary files
        for i in range(len(images)):
            temp_path = os.path.join(IMAGES_FOLDER, f"temp_{i}_{images[i].filename}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return collage_filename
        
    except Exception as e:
        print(f"Error creating image collage: {e}")
        raise e

def resize_image_to_fill(image, target_width, target_height):
    """Resize and crop image to fill target dimensions"""
    # Calculate scaling factors
    scale_x = target_width / image.width
    scale_y = target_height / image.height
    scale = max(scale_x, scale_y)  # Use larger scale to ensure coverage
    
    # Resize image
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Crop to target dimensions
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    return resized.crop((left, top, right, bottom))

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

@app.route('/extract_frame', methods=['POST'])
def extract_frame():
    try:
        data = request.get_json()
        video_filename = data.get('filename')
        timestamp = float(data.get('timestamp', 0))
        
        if not video_filename:
            return {'error': 'No filename provided'}, 400
            
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        if not os.path.exists(video_path):
            return {'error': 'Video file not found'}, 404
            
        # Extract frame using MoviePy
        clip = VideoFileClip(video_path)
        if timestamp >= clip.duration:
            timestamp = clip.duration - 0.1  # Stay within bounds
            
        frame = clip.get_frame(timestamp)
        clip.close()
        
        # Convert to PIL Image and save
        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame)
        
        # Generate unique filename for the frame
        base_name = os.path.splitext(video_filename)[0]
        frame_filename = f"{base_name}_frame_{int(timestamp*1000)}.png"
        frame_path = os.path.join(OUTPUT_FOLDER, frame_filename)
        
        img.save(frame_path)
        
        return {
            'success': True,
            'frame_filename': frame_filename,
            'download_url': url_for('download_clip', filename=frame_filename)
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
