import argparse
import os
import sys

import numpy as np
from datasets import Array2D, Dataset, Features, Sequence, Value, load_dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils


def process_dataset(data_root: str, output_dir: str, k: int = 8192) -> None:
    # Load dataset to get the order of images
    print("Loading dataset structure...")
    ds = load_dataset("laion/flux-ultra-stockimages", split="train")
    print(f"Dataset contains {len(ds)} images.")

    def generator():
        for item in tqdm(ds):
            key = item["__key__"]
            txt = item["txt"]
            # Construct full path
            image_path = os.path.join(data_root, f"{key}.jpg")

            try:
                if not os.path.exists(image_path):
                    # Yield padding for missing images
                    yield {
                        "txt": txt,
                        "fft_values": np.zeros((k, 2), dtype=np.float32),
                        "fft_positions": np.full((k, 2), -1.0, dtype=np.float32),
                        "original_shape": [0, 0],
                        "valid": False,
                    }
                    continue

                # Load and process image
                img = utils.load_image(image_path)
                gray_img = utils.to_grayscale(img)
                img_array = np.array(gray_img)
                shape = img_array.shape

                fft_res = utils.compute_fft(img_array)
                fft_norm = utils.normalize_fft(fft_res)
                values, positions = utils.get_top_frequencies(fft_norm, k=k)

                yield {
                    "txt": txt,
                    "fft_values": values.astype(np.float32),
                    "fft_positions": positions.astype(np.float32),
                    "original_shape": shape,
                    "valid": True,
                }

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                yield {
                    "txt": txt,
                    "fft_values": np.zeros((k, 2), dtype=np.float32),
                    "fft_positions": np.full((k, 2), -1.0, dtype=np.float32),
                    "original_shape": [0, 0],
                    "valid": False,
                }

    # Define the schema for the new dataset
    features = Features(
        {
            "txt": Value("string"),
            "fft_values": Array2D(shape=(k, 2), dtype="float32"),
            "fft_positions": Array2D(shape=(k, 2), dtype="float32"),
            "original_shape": Sequence(Value("int32"), length=2),
            "valid": Value("bool"),
        }
    )

    print("Processing dataset and saving to disk (this may take a while)...")
    # Create dataset from generator
    new_ds = Dataset.from_generator(generator, features=features)

    # Save to disk
    print(f"Saving dataset to {output_dir}...")
    new_ds.save_to_disk(output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images and extract top FFT frequencies into a HF Dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/ocean/projects/ees250005p/tho2/ImgFT/data/",
        help="Root directory containing the storage folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fft_dataset",
        help="Output directory for the HF dataset",
    )
    parser.add_argument(
        "--k", type=int, default=8192, help="Number of top frequencies to keep"
    )

    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.k)
