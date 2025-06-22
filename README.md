# Vehicle Detection

Il progetto di *crowd detection* mira a confrontare l’efficacia di diversi modelli di rilevamento e conteggio di persone in contesti affollati, utilizzando tre dataset di riferimento: **EAGLE**, **RAiVD** e **SODA**. Questi dataset offrono scenari diversificati — da immagini aeree urbane ad ambienti di sorveglianza a bassa risoluzione — permettendo una valutazione comparativa delle prestazioni dei modelli in termini di accuratezza, robustezza e generalizzazione. L'obiettivo è identificare i punti di forza e le criticità di ciascun approccio per ottimizzare le soluzioni di crowd monitoring in contesti reali.

Vengono confrontati tra loro modelli YOLO-based e modelli con backbone resnet50 a cui viene collegato un layer di regressione (di varie tipologie)

Scaricati i dataset (https://universe.roboflow.com/vehicledetection-wkkoq) è necessario portarli ad una forma comune per poter essre processati


# Setup Ambiente e Istruzioni per Esecuzione

## 1. Creazione dell'ambiente Python

```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

## 2. Installazione dei pacchetti

```bash
pip install -r requirements.txt
```

## 3. Conversione dataset da COCO a YOLO
```bash
python coco2yolo.py
```

## 4. Scelta del modello da testare

### Opzione A: Modelli YOLO

Per provare i modelli basati su YOLO:

- `coco2yolo_dataset_format.py` per avere un formato compatibile e poi lo script per il dataset

- `yolo11_RAIVD.ipynb` — RAIVD  
- `yolo11_EAGLE.ipynb` — EAGLE  
- `yolo11_SODA.ipynb` — SODA

Sono presenti sia col modello di base pre-trained di yolo e che un fine tuning sulla porzione di train del dataset (e quindi validazione e test sugli altri set)

### Opzione B: Modello ResNet50 per Regressione

Se si desidera provare un approccio alternativo basato su regressione con ResNet50 come backbone, usare il notebook:

- `backbone_resnet50_regression.ipynb`


