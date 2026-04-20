from flask import Flask, request, render_template
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER

app = Flask(__name__)

def allowed_file(filename):
    if '.' in filename:
        ext = filename.split('.')[-1].lower()
        if ext in ALLOWED_EXTENSIONS:
            return True
    return False

def add_gaussian_noise(image, strength):
    noise = np.random.normal(0, strength, image.shape)
    noisy = image + noise 
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_uniform_noise(image, strength):
    noise = np.random.uniform(-strength, strength, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_chisquare_noise(image, strength, k):
    noise = np.random.chisquare(k, image.shape)
    noise = noise - k
    noise = noise * strength / max(k, 1)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def exponential_noise(image, strenght):
    noise = np.random.exponential(strenght, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8) 

def gamma_noise(image, strength, k):
    noise = np.random.gamma(k, strength, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8) 

def weibull_noise(image, strength, k):
    noise = np.random.weibull(k, image.shape)
    noise = noise * strength / max(k, 1)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8) 

def apply_noise(image, noise_type, strength, k):
    if noise_type == "gaussian":
        img = add_gaussian_noise(image, strength)
        return img
    elif noise_type == "uniform":
        img = add_uniform_noise(image, strength)
        return img
    elif noise_type == "chisquare":
        img = add_chisquare_noise(image, strength, k)
        return img
    elif noise_type == "exponential":
        img = exponential_noise(image, strength)
        return img 
    elif noise_type == "gamma":
        img = gamma_noise(image, strength, k)
        return img
    elif noise_type == "weibull":
        img = weibull_noise(image, strength, k)
        return img
    else:
        return image


def selfmade_noise_filter(image):
    img = image.astype(np.float32)
    result = img.copy()
    h, w, c = img.shape
    for y in range(h - 1):
        for x in range(w - 1):
            for z in range(c):
                if img[y, x, z] > 3 * img[y, x + 1, z]:
                    result[y, x, z] = (img[y, x, z] + img[y, x + 1, z]) / 2
    return np.clip(result, 0, 255).astype(np.uint8)

# def median_filter(image):
#     img = image.astype(np.float32)
#     result = np.zeros_like(image)
#     h, w, c = img.shape
#     pad = 1 
#     padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
#     for y in range(h):
#         for x in range(w):
#             for z in range(c):
#                 window = padded[y:y+3, x:x+3, z]
#                 median_value = np.median(window)
#                 result[y, x, z] = median_value
#     return np.clip(result, 0, 255).astype(np.uint8)
 
def median_filter(image):
   return cv2.medianBlur(image, 5)


# def bilateral_filter(image, kernel_size=5, sigma_space=20, sigma_color=30):
#     h, w, c = image.shape
#     result = np.zeros_like(image)

#     for channel in range(c):
#         result[:, :, channel] = bilateral_filter(
#             image[:, :, channel],
#             kernel_size=kernel_size,
#             sigma_space=sigma_space,
#             sigma_color=sigma_color
#         )

#     return result

def bilaterial_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

    
def plot_histograms(image, noise_type, *args):
    plt.hist(image.flatten(), bins=256, range=(0, 255), log=True, color='#2563eb')
    
    if noise_type == "gaussian":
        plt.title("Гистограмма гаусса (нормальное распределение)")
    elif noise_type == "uniform":
        plt.title("Гистограмма равномерного распределения")
    elif noise_type == "chisquare":
        plt.title("Гистограмма хи-квадрат")
    elif noise_type == "exponential":
        plt.title("Гистограмма экспоненциального распределения")
    elif noise_type == "gamma":
        plt.title("Гистограмма гамма распределения")
    elif noise_type == "weibull":
        plt.title("Гистограмма вейбулла")
        
    plt.xlabel("Значение пикселя")
    plt.ylabel("Частота (логарифмическая шкала)")
    plt.savefig(f"static/uploads/{noise_type}_histogram.png")
    plt.close()


@app.route("/", methods=["GET", "POST"])
def index():
    original_path = None
    processed_path = None
    error_message = None
    histogram_path = None
    filtred_path = None
    median_path = None
    bilaterial_path = None

    if request.method == "POST":
        file = request.files.get("image")
        
        noise_type = request.form.get("noise_type")
        if noise_type is None:
            noise_type = "gaussian"
            
        strength = request.form.get("strength")
        if strength is None:
            strength = 100
        else:
            strength = float(strength)
            
        k = request.form.get("k")
        if k is None:
            k = 1
        else:
            k = float(k)

        if not file:
            error_message = "Файл не выбран."
        elif not allowed_file(file.filename):
            error_message = "Разрешены только файлы PNG, JPG и JPEG."
        else:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                error_message = "Не удалось прочитать изображение."
            else:

                processed = apply_noise(image, noise_type, strength, k)
                filtred = selfmade_noise_filter(processed)
                median = median_filter(processed)
                bilaterial = bilaterial_filter(processed)

                plot_histograms(processed, noise_type, strength, k)
                if noise_type == "gaussian":
                    original_filename = "original.png"
                    processed_filename = "gaussian.png"
                    histogram_filename ="gaussian_histogram.png"
                    filtred_filename = "guassian_after_selfmade_noise_filter.png"
                    median_filename = " gaussian_after_median_filter.png"
                    bilaterial_filename = "gaussian_after_bilaterial_filter.png"
                elif noise_type == "uniform":
                    original_filename = "original.png"
                    processed_filename = "uniform.png"
                    histogram_filename ="uniform_histogram.png"
                    filtred_filename = "uniform_after_selfmade_noise_filter.png"
                    median_filename = "uniform_after_median_filter.png"
                    bilaterial_filename = "uniform_after_bilaterial_filter.png"
                elif noise_type == "chisquare":
                    original_filename = "original.png"
                    processed_filename = "chisquare.png"
                    histogram_filename ="chisquare_histogram.png" 
                    filtred_filename = "chisquare_after_selfmade_noise_filter.png"
                    median_filename = "chisquare_after_median_filter.png"
                    bilaterial_filename = "chisquare_after_bilaterial_filter.png"
                elif noise_type == "exponential":
                    original_filename = "original.png"
                    processed_filename = "exponential.png"
                    histogram_filename ="exponential_histogram.png"
                    filtred_filename = "exponential_after_selfmade_noise_filter.png"
                    median_filename = "exponential_after_median_filter.png"
                    bilaterial_filename = "exponential_after_bilaterial_filter.png"
                elif noise_type == "gamma":
                    original_filename = "original.png"
                    processed_filename = "gamma.png"
                    histogram_filename ="gamma_histogram.png"
                    filtred_filename = "gamma_after_selfmade_noise_filter.png"
                    median_filename = "gamma_after_median_filter.png"
                    bilaterial_filename = "gamma_after_bilaterial_filter.png"
                elif noise_type == "weibull":
                    original_filename = "original.png"
                    processed_filename = "weibull.png"
                    histogram_filename ="weibull_histogram.png"
                    filtred_filename = "weibull_after_selfmade_noise_filter.png"
                    median_filename = "weibull_after_median_filter.png"
                    bilaterial_filename = "weibull_after_bilaterial_filter.png"


                original_full_path = os.path.join(UPLOAD_FOLDER, original_filename)
                processed_full_path = os.path.join(UPLOAD_FOLDER, processed_filename)
                histogram__full_path = os.path.join(UPLOAD_FOLDER, histogram_filename)
                filtred_full_path = os.path.join(UPLOAD_FOLDER, filtred_filename)
                median_full_path = os.path.join(UPLOAD_FOLDER, median_filename)
                bilaterial_full_path = os.path.join(UPLOAD_FOLDER, bilaterial_filename)


                cv2.imwrite(original_full_path, image)
                cv2.imwrite(processed_full_path, processed)
                cv2.imwrite(filtred_full_path, filtred)
                cv2.imwrite(median_full_path, median)
                cv2.imwrite(bilaterial_full_path, bilaterial)

                histogram_path = histogram__full_path.replace("\\", "/")  
                original_path = original_full_path.replace("\\", "/")
                processed_path = processed_full_path.replace("\\", "/")
                filtred_path = filtred_full_path.replace("\\", "/")
                median_path = median_full_path.replace("\\", "/")
                bilaterial_path = bilaterial_full_path.replace("\\", "/")

    return render_template(
        "index.html",
        original_path=original_path,
        processed_path=processed_path,
        error_message=error_message,
        histogram_path=histogram_path,
        filtred_path=filtred_path,
        median_path=median_path,
        bilaterial_path=bilaterial_path
    )


if __name__ == "__main__":
    app.run(debug=True)