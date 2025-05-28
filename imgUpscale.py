import cv2

def upscale_image(image_path, scale=2):
    # Вчитување на слика
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Сликата не може да се вчита.")
        return

    # Пресметка на нова резолуција
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    # Зголемување со интерполација (Lanczos е многу добар за квалитет)
    upscaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # Прикажи и зачувај
    cv2.imshow("Upscaled Image", upscaled)
    cv2.imwrite("upscaled_output.jpg", upscaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✔️ Сликата е зголемена и зачувана како upscaled_output.jpg")

# Тест
upscale_image("group2.jpg", scale=3)