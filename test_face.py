import cv2
from insightface.app import FaceAnalysis

# Inicializar detector
print("Cargando modelo...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Modelo cargado!")

# Abrir cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

print("Capturando imagen...")

ret, frame = cap.read()
if ret:
    # Detectar caras
    faces = app.get(frame)
    
    print(f"Caras detectadas: {len(faces)}")
    
    for i, face in enumerate(faces):
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        print(f"Cara {i+1}: bbox={box}, embedding shape={face.embedding.shape}")
    
    # Guardar imagen
    cv2.imwrite('/data/resultado.jpg', frame)
    print("Imagen guardada en /data/resultado.jpg")

cap.release()
