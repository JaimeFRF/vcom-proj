import os
import cv2
from PIL import Image

def is_image_valid_opencv(path):
    img = cv2.imread(path)
    return img is not None

def is_image_valid_pil(path):
    try:
        with Image.open(path) as img:
            img.verify()  # Verify image file integrity
        return True
    except Exception:
        return False

def check_images_integrity(root_dir, use_pil=False):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    corrupted_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
                print(f"Checking {filename} in {dirpath}... ")
                full_path = os.path.join(dirpath, filename)

                if use_pil:
                    valid = is_image_valid_pil(full_path)
                else:
                    valid = is_image_valid_opencv(full_path)

                if not valid:
                    corrupted_files.append(full_path)

    print(f"\nChecked all images in '{root_dir}'")
    if corrupted_files:
        print(f"❌ Found {len(corrupted_files)} corrupted or unreadable images:")
        for path in corrupted_files:
            print("  -", path)
    else:
        print("✅ All images are valid.")

    return corrupted_files

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check image integrity recursively.")
    parser.add_argument("path", help="Root directory to check")
    parser.add_argument("--pil", action="store_true", help="Use PIL for additional format checks")
    args = parser.parse_args()

    check_images_integrity(args.path, use_pil=args.pil)