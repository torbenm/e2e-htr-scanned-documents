def copy(images, num):
    new_images = []
    for image in images:
        for i in range(num):
            new_images.append(image.copy())
    images.extend(new_images)
    return images
