import cv2
import numpy as np
import json
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# Configuración
DB_FILE = '/app/empleados.json'
LOGS_FILE = '/app/asistencia_log.json'
THRESHOLD = 0.4

# Inicializar detector
print("Cargando modelo...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Modelo cargado!")

def cargar_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            for emp in data:
                emp['embedding'] = np.array(emp['embedding'])
            return data
    return []

def guardar_db(empleados):
    data = []
    for emp in empleados:
        data.append({
            'nombre': emp['nombre'],
            'embedding': emp['embedding'].tolist()
        })
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

def registrar_log(nombre, tipo):
    log = {'nombre': nombre, 'tipo': tipo, 'timestamp': datetime.now().isoformat()}
    logs = []
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            logs = json.load(f)
    logs.append(log)
    with open(LOGS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    return log

def comparar_embedding(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def reconocer(embedding, empleados):
    mejor_match = None
    mejor_score = 0
    for emp in empleados:
        score = comparar_embedding(embedding, emp['embedding'])
        if score > mejor_score:
            mejor_score = score
            mejor_match = emp['nombre']
    if mejor_score > THRESHOLD:
        return mejor_match, mejor_score
    return None, mejor_score

def main():
    empleados = cargar_db()
    print(f"Empleados registrados: {len(empleados)}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return
    
    print("Controles:")
    print("  'r' - Registrar nueva persona")
    print("  'q' - Salir")
    
    detectados_recientes = {}
    ultimo_log = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detectar caras
        faces = app.get(frame)
        
        for face in faces:
            box = face.bbox.astype(int)
            nombre, score = reconocer(face.embedding, empleados)
            
            if nombre:
                color = (0, 255, 0)  # Verde
                label = f"{nombre} ({score:.2f})"
                
                ahora = datetime.now()
                if nombre not in detectados_recientes or \
                   (ahora - detectados_recientes[nombre]).seconds > 30:
                    detectados_recientes[nombre] = ahora
                    log = registrar_log(nombre, "ENTRADA")
                    ultimo_log = f"{nombre} - ENTRADA - {ahora.strftime('%H:%M:%S')}"
            else:
                color = (0, 0, 255)  # Rojo
                label = "Desconocido"
            
            # Dibujar rectángulo y nombre
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info en pantalla
        cv2.putText(frame, f"Empleados: {len(empleados)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Ultimo: {ultimo_log}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "R=Registrar | Q=Salir", (10, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Sistema de Asistencia', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Registrar nueva persona
            print("\n--- REGISTRO ---")
            nombre = input("Nombre del empleado: ").strip()
            if nombre:
                print("Mira a la cámara...")
                ret, frame = cap.read()
                if ret:
                    faces = app.get(frame)
                    if len(faces) == 1:
                        empleados.append({
                            'nombre': nombre,
                            'embedding': faces[0].embedding
                        })
                        guardar_db(empleados)
                        print(f"✓ {nombre} registrado!")
                    else:
                        print(f"Error: {len(faces)} caras detectadas")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
