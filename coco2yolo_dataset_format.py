import json
import os
from tqdm import tqdm

# Funzione per convertire le annotazioni COCO in formato YOLO
def convert_coco_to_yolo(coco_json_file, images_dir, output_dir, class_names):
    # Carica il file JSON delle annotazioni COCO
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Crea un dizionario per mappare gli ID delle categorie ai loro nomi
    category_id_map = {category['id']: category['name'] for category in coco_data['categories']}

    # Crea una mappa per la categoria in formato YOLO (starta da 0)
    class_id_map = {name: idx for idx, name in enumerate(class_names)}

    # Crea un dizionario che tiene traccia delle immagini
    images = {image['id']: image['file_name'] for image in coco_data['images']}

    # Itera su ogni annotazione e crea i file YOLO
    #for ann in coco_data['annotations']:
    for ann in tqdm(coco_data['annotations'], desc="Converting annotations", unit="annotation"):
        # Ottieni l'immagine
        img_id = ann['image_id']
        image_name = images[img_id]

        # Ottieni le dimensioni dell'immagine
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_width = img_info['width']
        img_height = img_info['height']

        # Estrai le informazioni del bounding box
        x, y, w, h = ann['bbox']

        # Normalizza le coordinate del bounding box
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        # Ottieni il class ID (e.g. "vehicle" -> 0, "person" -> 1)
        class_name = category_id_map[ann['category_id']]
        class_id = class_id_map[class_name]

        # Crea il nome del file YOLO per l'immagine
        yolo_annotation_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")

        # Scrivi l'annotazione nel file YOLO (se il file esiste già, lo aggiorniamo)
        with open(yolo_annotation_file, 'a') as file:
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Conversione completata. Le annotazioni YOLO sono state salvate in: {output_dir}")

# Funzione principale che converte tutte le cartelle train, valid, test
def convert_all_folders_to_yolo(main_folder):
    for split in ['train', 'valid', 'test']:
        print(f"Converting {split} dataset...")
        coco_json_file = os.path.join(main_folder, split, '_annotations.coco.json')
        class_names = extract_class_names(coco_json_file)  # Estrai automaticamente le categorie
        images_dir = os.path.join(main_folder, split)  # Le immagini sono direttamente in 'train', 'valid' e 'test'
        output_dir = images_dir  # Le annotazioni vanno direttamente nella cartella delle immagini

        # Converte il file COCO in YOLO per ciascuna delle cartelle
        convert_coco_to_yolo(coco_json_file, images_dir, output_dir, class_names)

# Funzione per estrarre le categorie dal file COCO
def extract_class_names(coco_json_file):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)
    class_names = [category['name'] for category in coco_data['categories']]
    return class_names

# Funzione per creare il file YAML per YOLO
def create_yaml(dataset_dir, class_names):
    yaml_file = os.path.join(dataset_dir, 'dataset.yaml')

    # Creazione dei percorsi per le immagini
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

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

# Percorso della cartella principale
main_folder = 'C:/Users/conca/Documents/progetti/vehicle_detection/EAGLE'  # Sostituisci con il percorso della tua cartella principale (es. 'SODA')
class_names = ['vehicle', 'object']  # Sostituisci con il nome delle categorie del tuo dataset

# Esegui la conversione per tutte le cartelle
convert_all_folders_to_yolo(main_folder)

# Creazione del file YAML per YOLO
create_yaml(main_folder, class_names)
