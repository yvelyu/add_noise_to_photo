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
                plot_histograms(processed, noise_type, strength, k)
                if noise_type == "gaussian":
                    original_filename = "original.png"
                    processed_filename = "gaussian.png"
                    histogram_filename ="gaussian_histogram.png"
                elif noise_type == "uniform":
                    original_filename = "original.png"
                    processed_filename = "uniform.png"
                    histogram_filename ="uniform_histogram.png"
                elif noise_type == "chisquare":
                    original_filename = "original.png"
                    processed_filename = "chisquare.png"
                    histogram_filename ="chisquare_histogram.png"
                elif noise_type == "exponential":
                    original_filename = "original.png"
                    processed_filename = "exponential.png"
                    histogram_filename ="exponential_histogram.png"
                elif noise_type == "gamma":
                    original_filename = "original.png"
                    processed_filename = "gamma.png"
                    histogram_filename ="gamma_histogram.png"
                elif noise_type == "weibull":
                    original_filename = "original.png"
                    processed_filename = "weibull.png"
                    histogram_filename ="weibull_histogram.png"

                original_full_path = os.path.join(UPLOAD_FOLDER, original_filename)
                processed_full_path = os.path.join(UPLOAD_FOLDER, processed_filename)
                histogram__full_path = os.path.join(UPLOAD_FOLDER, histogram_filename)


                cv2.imwrite(original_full_path, image)
                cv2.imwrite(processed_full_path, processed)

                histogram_path = histogram__full_path.replace("\\", "/")  
                original_path = original_full_path.replace("\\", "/")
                processed_path = processed_full_path.replace("\\", "/")

    return render_template(
        "index.html",
        original_path=original_path,
        processed_path=processed_path,
        error_message=error_message,
        histogram_path=histogram_path
    )


if __name__ == "__main__":
    app.run(debug=True)