import json
import os
from tqdm import tqdm

main_folder = 'SODA'  # ← sostituisci con la tua cartella

def convert_coco_to_yolo(coco_json_file, images_dir, output_dir):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Estrai i nomi delle classi in ordine crescente di ID COCO
    categories_sorted = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories_sorted]

    # Mappa category_id COCO → class_id YOLO (0-based)
    category_id_to_yolo_id = {
        cat['id']: idx for idx, cat in enumerate(categories_sorted)
    }

    # Mappa immagini
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    sizes = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    for ann in tqdm(coco_data['annotations'], desc="Converting annotations"):
        img_id = ann['image_id']
        image_name = images[img_id]
        img_width, img_height = sizes[img_id]

        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        class_id = category_id_to_yolo_id[ann['category_id']]

        yolo_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(yolo_file, 'a') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Annotazioni YOLO salvate in: {output_dir}")
    return class_names


def create_yaml(dataset_dir, class_names):
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    yaml_content = f"""train: {os.path.join(dataset_dir, 'train')}
val: {os.path.join(dataset_dir, 'valid')}
test: {os.path.join(dataset_dir, 'test')}

nc: {len(class_names)}
names: {class_names}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"File YAML creato: {yaml_path}")


def convert_all_folders_to_yolo(main_folder):
    for split in ['train', 'valid', 'test']:
        coco_json = os.path.join(main_folder, split, '_annotations.coco.json')
        if not os.path.exists(coco_json):
            continue
        output_dir = os.path.join(main_folder, split)
        print(f"Convertendo {split}...")
        class_names = convert_coco_to_yolo(coco_json, output_dir, output_dir)
    create_yaml(main_folder, class_names)


# ESECUZIONE

convert_all_folders_to_yolo(main_folder)
