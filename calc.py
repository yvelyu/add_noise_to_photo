import os
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

IMAGE_DIR = "static/uploads"
CSV_PATH = "metrics.csv"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

KNOWN_FILTERS = {
    "median_filter",
    "selfmade_noise_filter",
    "bilaterial_filter"
}


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def calc_mse(img1, img2):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.mean(diff ** 2)


def calc_psnr(img1, img2):
    mse_value = calc_mse(img1, img2)
    if mse_value == 0:
        return float("inf")
    return 10 * np.log10((255 ** 2) / mse_value)


def calc_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)


def parse_file_info(filename):
    name, _ = os.path.splitext(filename)

    if name == "original":
        return {
            "stage": "original",
            "noise_type": "",
            "filter_type": ""
        }

    if "_after_" in name:
        noise_type, filter_type = name.split("_after_", 1)

        if filter_type in KNOWN_FILTERS:
            return {
                "stage": "filtered",
                "noise_type": noise_type,
                "filter_type": filter_type
            }

    return {
        "stage": "noise",
        "noise_type": name,
        "filter_type": ""
    }


def main():
    files = [f for f in os.listdir(IMAGE_DIR) if is_image_file(f)]

    original_candidates = [f for f in files if os.path.splitext(f)[0] == "original"]

    if not original_candidates:
        print("Ошибка: файл original.png / original.jpg / original.jpeg не найден.")
        return

    original_filename = original_candidates[0]
    original_path = os.path.join(IMAGE_DIR, original_filename)

    original_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
    if original_img is None:
        print("Ошибка: не удалось прочитать оригинальное изображение.")
        return

    rows = []

    for filename in files:
        if filename == original_filename:
            continue

        path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Пропуск: {filename} не удалось прочитать.")
            continue

        if img.shape[:2] != original_img.shape[:2]:
            img = cv2.resize(img, (original_img.shape[1], original_img.shape[0]))

        info = parse_file_info(filename)

        mse_value = calc_mse(original_img, img)
        psnr_value = calc_psnr(original_img, img)
        ssim_value = calc_ssim(original_img, img)

        rows.append({
            "original_file": original_filename,
            "compared_file": filename,
            "stage": info["stage"],
            "noise_type": info["noise_type"],
            "filter_type": info["filter_type"],
            "mse": round(float(mse_value), 4),
            "psnr": round(float(psnr_value), 4) if np.isfinite(psnr_value) else "inf",
            "ssim": round(float(ssim_value), 6)
        })

    rows.sort(key=lambda x: (x["noise_type"], x["stage"], x["filter_type"]))

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_file",
                "compared_file",
                "stage",
                "noise_type",
                "filter_type",
                "mse",
                "psnr",
                "ssim"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Готово: {CSV_PATH}")


if __name__ == "__main__":
    main()