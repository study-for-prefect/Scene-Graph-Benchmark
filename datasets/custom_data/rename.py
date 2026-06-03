import os


def rename_images(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    if not os.path.exists(directory):
        print(f"错误：路径 {directory} 不存在。")
        return

    files = sorted(os.listdir(directory))

    count = 1
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            old_path = os.path.join(directory, filename)
            new_filename = f"{count}{ext}"
            new_path = os.path.join(directory, new_filename)

            while os.path.exists(new_path):
                count += 1
                new_filename = f"{count}{ext}"
                new_path = os.path.join(directory, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {new_filename}")
                count += 1
            except Exception as e:
                print(f"重命名失败 {filename}: {e}")


if __name__ == "__main__":
    target_directory = "/home/wxm/dataset/building_block"
    rename_images(target_directory)