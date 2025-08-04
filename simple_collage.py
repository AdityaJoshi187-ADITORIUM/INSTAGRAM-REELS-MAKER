from flask import Flask, request, render_template, send_from_directory, url_for
from PIL import Image
import os
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
IMAGES_FOLDER = 'images'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        images = request.files.getlist('collage_images')
        layout = request.form.get('collage_layout', 'grid_2x2')
        aspect_ratio = request.form.get('collage_aspect_ratio', '1:1')
        background_color = request.form.get('background_color', '#ffffff')
        
        if not images or len(images) < 2:
            return render_template('simple_collage.html', error="Please upload at least 2 images for collage.")
        
        collage_filename = create_simple_collage(images, layout, aspect_ratio, background_color)
        return render_template('simple_collage.html', collage_result=collage_filename)

    return render_template('simple_collage.html')

def create_simple_collage(images, layout='grid_2x2', aspect_ratio='1:1', background_color='#ffffff'):
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

def resize_image_to_fit(image, target_width, target_height, background_color='#ffffff'):
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
    
    # Create a new image with target dimensions and specified background
    result = Image.new('RGB', (target_width, target_height), background_color)
    
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

@app.route('/output/<filename>')
def download_clip(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)

@app.route('/editor', methods=['GET', 'POST'])
def image_editor():
    if request.method == 'POST':
        # Handle the drag-and-drop editor
        if 'save_collage' in request.form:
            # Get the canvas data and save the final image
            canvas_data = request.form.get('canvas_data')
            if canvas_data:
                # Remove the data URL prefix
                if canvas_data.startswith('data:image/png;base64,'):
                    canvas_data = canvas_data.split(',')[1]
                
                import base64
                from io import BytesIO
                
                # Decode base64 and save
                image_data = base64.b64decode(canvas_data)
                img = Image.open(BytesIO(image_data))
                
                timestamp = int(time.time())
                collage_filename = f"custom_collage_{timestamp}.jpg"
                collage_path = os.path.join(OUTPUT_FOLDER, collage_filename)
                img.save(collage_path, 'JPEG', quality=95)
                
                return render_template('image_editor.html', collage_result=collage_filename)
        
        # Handle image uploads for the editor
        images = request.files.getlist('editor_images')
        if images:
            uploaded_images = []
            for i, image_file in enumerate(images):
                if image_file and image_file.filename:
                    # Save uploaded image
                    image_path = os.path.join(IMAGES_FOLDER, f"editor_{i}_{image_file.filename}")
                    image_file.save(image_path)
                    uploaded_images.append({
                        'filename': f"editor_{i}_{image_file.filename}",
                        'path': image_path
                    })
            return render_template('image_editor.html', uploaded_images=uploaded_images)
    
    return render_template('image_editor.html')

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True) 