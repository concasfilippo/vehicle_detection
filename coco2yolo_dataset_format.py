import json
import os
from tqdm import tqdm

import os

###### NOME CARTELLA
main_folder = 'SODA'  # Percorso cartella principale (es. 'SODA')




def decrement_yolo_class_ids(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            updated_lines = []

            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0]) - 1  # decrementa di 1
                        if class_id < 0:
                            raise ValueError(f"class_id negativo in {filename}")
                        updated_line = ' '.join([str(class_id)] + parts[1:])
                        updated_lines.append(updated_line)
                    except ValueError:
                        print(f"⚠️  Ignorata riga non valida in {filename}: {line.strip()}")

            with open(file_path, 'w') as f:
                for line in updated_lines:
                    f.write(line + '\n')

    print(f"✅ Class_id decrementati in tutti i file .txt in {folder_path}")


# Funzione per convertire le annotazioni COCO in formato YOLO
def convert_coco_to_yolo(coco_json_file, images_dir, output_dir, category_id_to_yolo_id):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    images = {img['id']: img['file_name'] for img in coco_data['images']}
    image_sizes = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    for ann in tqdm(coco_data['annotations'], desc="Converting annotations", unit=" annotation"):
        img_id = ann['image_id']
        image_name = images[img_id]
        img_width, img_height = image_sizes[img_id]

        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        # Finalmente corretto
        class_id = category_id_to_yolo_id[ann['category_id']]

        yolo_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(yolo_file, 'a') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Conversione completata. Le annotazioni YOLO sono state salvate in: {output_dir}")






# Funzione principale che converte tutte le cartelle train, valid, test
def convert_all_folders_to_yolo(main_folder):
    for split in ['train', 'valid', 'test']:
        print(f"Converting {split} dataset...")
        coco_json_file = os.path.join(main_folder, split, '_annotations.coco.json')

        # Estrai le categorie con id e nome
        categories = extract_class_names(coco_json_file)
        class_names = [cat['name'] for cat in categories]
        category_id_to_yolo_id = {cat['id']: idx for idx, cat in enumerate(categories)}
        print(category_id_to_yolo_id)

        images_dir = os.path.join(main_folder, split)
        output_dir = images_dir

        convert_coco_to_yolo(coco_json_file, images_dir, output_dir, category_id_to_yolo_id)

        decrement_yolo_class_ids(main_folder + "/" + split)


# Funzione per estrarre le categorie dal file COCO
def extract_class_names(coco_json_file):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)
    categories_sorted = sorted(coco_data['categories'], key=lambda x: x['id'])
    return categories_sorted  # ← Ritorna oggetti completi, non solo nomi


# Funzione per creare il file YAML per YOLO
def create_yaml(dataset_dir, class_names):
    yaml_file = os.path.join(dataset_dir, 'dataset.yaml')

    # Creazione dei percorsi per le immagini
    train_dir = os.path.join("", 'train')
    val_dir = os.path.join("", 'valid')
    test_dir = os.path.join("", 'test')

    # Costruisci il contenuto del file YAML
    yaml_content = f"""
    train: {train_dir}
    val: {val_dir}
    test: {test_dir}

    nc: {len(class_names)}  # Numero di categorie
    names: {class_names}  # Lista delle categorie
    """

    # Scrivi il file YAML
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    print(f"File YAML creato: {yaml_file}")


class_names = ['vehicles-Z6jE-iVPc', 'class_2', 'class_3']  # Nome delle categorie del dataset

# Esegui la conversione per tutte le cartelle
convert_all_folders_to_yolo(main_folder)

# Creazione del file YAML per YOLO
create_yaml(main_folder, class_names)

import os
import shutil

def fix_yolo_structure(base_dir):
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base_dir, split, 'images')
        lbl_dir = os.path.join(base_dir, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for f in os.listdir(os.path.join(base_dir, split)):
            full_path = os.path.join(base_dir, split, f)
            if f.endswith(('.jpg', '.png')):
                shutil.move(full_path, os.path.join(img_dir, f))
            elif f.endswith('.txt'):
                shutil.move(full_path, os.path.join(lbl_dir, f))

# Esempio d’uso
fix_yolo_structure(main_folder)

