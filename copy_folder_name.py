import os
import tkinter as tk
from tkinter import filedialog, messagebox

def select_folder(title):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

def clone_full_structure(src_root, dst_root):
    if not os.path.exists(src_root):
        print("❌ Thư mục nguồn không tồn tại")
        return

    for root, dirs, files in os.walk(src_root):
        # Lấy đường dẫn tương đối
        rel_path = os.path.relpath(root, src_root)
        dst_path = os.path.join(dst_root, rel_path)

        # Tạo folder hiện tại
        os.makedirs(dst_path, exist_ok=True)

        # Tạo các folder con
        for d in dirs:
            new_dir = os.path.join(dst_path, d)
            os.makedirs(new_dir, exist_ok=True)
            print(f"✅ {os.path.join(rel_path, d)}")

def main():
    print("👉 Chọn thư mục nguồn")
    src = select_folder("Chọn thư mục nguồn")

    if not src:
        return

    print("👉 Chọn thư mục đích")
    dst = select_folder("Chọn thư mục đích")

    if not dst:
        return

    clone_full_structure(src, dst)

    messagebox.showinfo("Hoàn tất", "Đã clone toàn bộ structure!")

if __name__ == "__main__":
    main()