import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64
import torchvision.transforms as T
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import argparse

# Import your models - adjust path if needed
from models import PointUNet

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)


class SegmentationModel:
    def __init__(self, model_path, device=None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = PointUNet(n_channels=3, n_classes=3).to(self.device)
        self.load_model(model_path)
        self.model.eval()

        # Set up image preprocessing
        self.img_size = (256, 256)
        self.transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])

        # Set class colors for visualization
        self.class_colors = {
            0: [0, 0, 0],  # Background
            1: [255, 0, 0],  # Cat (red)
            2: [0, 0, 255]  # Dog (blue)
        }

    def load_model(self, model_path):
        """Load the trained model from checkpoint"""
        try:
            # Try with weights_only=False first (for PyTorch 2.6+ compatibility)
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Older PyTorch versions don't have weights_only parameter
                checkpoint = torch.load(model_path, map_location=self.device)

            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model state from checkpoint dictionary")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded model state directly")

            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def preprocess_image(self, image_data):
        """Process image data from request"""
        # Convert base64 to PIL image
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')

        # Store original size for later
        orig_size = image.size

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        return image_tensor, orig_size, image

    def generate_heatmap(self, point, img_size, heatmap_type="point", scribble_points=None, box_points=None,
                         text_prompt=None):
        """Generate appropriate heatmap based on prompt type"""
        if heatmap_type == "point":
            y, x = point
            heatmap = np.zeros(img_size, dtype=np.float32)

            # Generate 2D Gaussian
            sigma = min(img_size) / 16  # Adaptive sigma based on image size
            y_grid, x_grid = np.ogrid[:img_size[0], :img_size[1]]
            heatmap = np.exp(-((y_grid - y) ** 2 + (x_grid - x) ** 2) / (2 * sigma ** 2))

        elif heatmap_type == "scribble" and scribble_points:
            heatmap = np.zeros(img_size, dtype=np.float32)

            # Generate individual heatmaps for each point and combine them
            for point in scribble_points:
                y, x = point
                sigma = min(img_size) / 32  # Smaller sigma for scribbles
                y_grid, x_grid = np.ogrid[:img_size[0], :img_size[1]]
                point_heatmap = np.exp(-((y_grid - y) ** 2 + (x_grid - x) ** 2) / (2 * sigma ** 2))
                heatmap = np.maximum(heatmap, point_heatmap)

            # Add lines between adjacent points to ensure continuity
            if len(scribble_points) > 1:
                for i in range(len(scribble_points) - 1):
                    pt1 = scribble_points[i]
                    pt2 = scribble_points[i + 1]

                    line_mask = np.zeros(img_size, dtype=np.uint8)
                    cv2.line(line_mask, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), 1, thickness=2)
                    line_mask = cv2.GaussianBlur(line_mask.astype(np.float32), (5, 5), 0)
                    heatmap = np.maximum(heatmap, line_mask)

            # Normalize the heatmap
            heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

        elif heatmap_type == "box" and box_points and len(box_points) == 2:
            heatmap = np.zeros(img_size, dtype=np.float32)

            # Extract box coordinates
            (y1, x1), (y2, x2) = box_points
            y_min, y_max = min(y1, y2), max(y1, y2)
            x_min, x_max = min(x1, x2), max(x1, x2)

            # Create binary mask for the box region
            heatmap[int(y_min):int(y_max), int(x_min):int(x_max)] = 1.0

            # Apply Gaussian blur for smoother transition
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

        elif heatmap_type == "text" and text_prompt:
            heatmap = np.zeros(img_size, dtype=np.float32)
            text = text_prompt.lower().strip()

            if "cat" in text:
                # Create a center-biased heatmap that's horizontally elongated (cat-like)
                y_center, x_center = img_size[0] // 2, img_size[1] // 2
                y_grid, x_grid = np.ogrid[:img_size[0], :img_size[1]]
                sigma_y = img_size[0] / 5
                sigma_x = img_size[1] / 4
                heatmap = np.exp(-((y_grid - y_center) ** 2 / (2 * sigma_y ** 2) +
                                   (x_grid - x_center) ** 2 / (2 * sigma_x ** 2)))

            elif "dog" in text:
                # Create a larger, more spread heatmap that's less focused (dog-like)
                y_center, x_center = img_size[0] // 2, img_size[1] // 2
                y_grid, x_grid = np.ogrid[:img_size[0], :img_size[1]]
                sigma_y = img_size[0] / 3
                sigma_x = img_size[1] / 3
                # Slightly offset from center
                x_offset = img_size[1] / 8
                heatmap = np.exp(-((y_grid - y_center) ** 2 / (2 * sigma_y ** 2) +
                                   (x_grid - (x_center + x_offset)) ** 2 / (2 * sigma_x ** 2)))

            else:
                # Default heatmap for other text
                y_center, x_center = img_size[0] // 2, img_size[1] // 2
                y_grid, x_grid = np.ogrid[:img_size[0], :img_size[1]]
                sigma = min(img_size) / 4
                heatmap = np.exp(-((y_grid - y_center) ** 2 + (x_grid - x_center) ** 2) / (2 * sigma ** 2))
        else:
            # Default to center point if no valid prompt
            y_center, x_center = img_size[0] // 2, img_size[1] // 2
            heatmap = np.zeros(img_size, dtype=np.float32)
            y_grid, x_grid = np.ogrid[:img_size[0], :img_size[1]]
            sigma = min(img_size) / 4
            heatmap = np.exp(-((y_grid - y_center) ** 2 + (x_grid - x_center) ** 2) / (2 * sigma ** 2))

        return torch.FloatTensor(heatmap).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image_tensor, heatmap_tensor):
        """Generate segmentation prediction"""
        # Move heatmap to device
        heatmap_tensor = heatmap_tensor.unsqueeze(0).to(self.device)

        # Run inference
        output = self.model(image_tensor, heatmap_tensor)

        # Get class predictions
        pred = torch.argmax(output, dim=1)[0].cpu().numpy()

        return pred

    def colorize_mask(self, mask):
        """Convert class indices to RGB color mask"""
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        for class_idx, color in self.class_colors.items():
            rgb_mask[mask == class_idx] = color

        return rgb_mask

    def process_request(self, image_data, prompt_data):
        """Process a segmentation request from the web interface"""
        # Preprocess image
        image_tensor, orig_size, orig_image = self.preprocess_image(image_data)

        # Get prompt type and data
        prompt_type = prompt_data.get('type', 'point')

        # Generate appropriate heatmap based on prompt type
        if prompt_type == 'point':
            # Scale point from original to model input size
            point = prompt_data.get('point')
            if point:
                x, y = point
                # Convert to y, x order for the model
                orig_w, orig_h = orig_size
                x_scaled = int((x / orig_w) * self.img_size[1])
                y_scaled = int((y / orig_h) * self.img_size[0])
                scaled_point = (y_scaled, x_scaled)

                heatmap = self.generate_heatmap(scaled_point, self.img_size)
            else:
                # Default center point
                center_point = (self.img_size[0] // 2, self.img_size[1] // 2)
                heatmap = self.generate_heatmap(center_point, self.img_size)

        elif prompt_type == 'scribble':
            # Scale all scribble points
            scribble_points = prompt_data.get('points', [])
            scaled_points = []

            for point in scribble_points:
                x, y = point
                # Convert to y, x order for the model
                orig_w, orig_h = orig_size
                x_scaled = int((x / orig_w) * self.img_size[1])
                y_scaled = int((y / orig_h) * self.img_size[0])
                scaled_points.append((y_scaled, x_scaled))

            heatmap = self.generate_heatmap(None, self.img_size,
                                            heatmap_type="scribble",
                                            scribble_points=scaled_points)

        elif prompt_type == 'box':
            # Scale both box points
            box_points = prompt_data.get('points', [])
            scaled_points = []

            for point in box_points:
                x, y = point
                # Convert to y, x order for the model
                orig_w, orig_h = orig_size
                x_scaled = int((x / orig_w) * self.img_size[1])
                y_scaled = int((y / orig_h) * self.img_size[0])
                scaled_points.append((y_scaled, x_scaled))

            heatmap = self.generate_heatmap(None, self.img_size,
                                            heatmap_type="box",
                                            box_points=scaled_points)

        elif prompt_type == 'text':
            text = prompt_data.get('text', '')
            heatmap = self.generate_heatmap(None, self.img_size,
                                            heatmap_type="text",
                                            text_prompt=text)
        else:
            # Default to center point
            center_point = (self.img_size[0] // 2, self.img_size[1] // 2)
            heatmap = self.generate_heatmap(center_point, self.img_size)

        # Generate prediction
        mask = self.predict(image_tensor, heatmap)

        # Colorize mask
        colored_mask = self.colorize_mask(mask)

        # Resize mask to match original image size
        colored_mask_resized = cv2.resize(colored_mask, (orig_size[0], orig_size[1]), interpolation=cv2.INTER_NEAREST)

        # Create blended visualization
        orig_np = np.array(orig_image)
        alpha = 0.5
        blended = cv2.addWeighted(orig_np, 1 - alpha, colored_mask_resized, alpha, 0)

        # Convert result to base64 for sending back to client
        blended_pil = Image.fromarray(blended)
        buffered = io.BytesIO()
        blended_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            'segmentation_image': img_str
        }


# Global model instance
model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/segment', methods=['POST'])
def segment():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Get image data
        image_data_uri = request.json['image']
        if ';base64,' in image_data_uri:
            _, image_data_base64 = image_data_uri.split(';base64,')
            image_data = base64.b64decode(image_data_base64)
        else:
            return jsonify({'error': 'Invalid image format'}), 400

        # Get prompt data
        prompt_data = request.json.get('prompt', {'type': 'point'})

        # Process the request
        result = model.process_request(image_data, prompt_data)

        return jsonify(result)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    global model

    parser = argparse.ArgumentParser(description="Interactive Pet Segmentation Web App")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained PointUNet model weights")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")

    args = parser.parse_args()

    # Initialize model
    model = SegmentationModel(args.model_path, args.device)

    # Run server
    app.run(debug=args.debug, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()