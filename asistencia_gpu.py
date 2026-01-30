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

print("Cargando modelo con GPU...")
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
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
    data = [{'nombre': e['nombre'], 'embedding': e['embedding'].tolist()} for e in empleados]
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

def comparar_embedding(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def reconocer(embedding, empleados):
    mejor_match, mejor_score = None, 0
    for emp in empleados:
        score = comparar_embedding(embedding, emp['embedding'])
        if score > mejor_score:
            mejor_score, mejor_match = score, emp['nombre']
    return (mejor_match, mejor_score) if mejor_score > THRESHOLD else (None, mejor_score)

def main():
    empleados = cargar_db()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Empleados: {len(empleados)} | R=Registrar | Q=Salir")
    
    detectados = {}
    prev_time = datetime.now()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        faces = app.get(frame)
        
        # Calcular FPS
        now = datetime.now()
        fps = 1 / max((now - prev_time).total_seconds(), 0.001)
        prev_time = now
        
        for face in faces:
            box = face.bbox.astype(int)
            nombre, score = reconocer(face.embedding, empleados)
            
            if nombre:
                color, label = (0, 255, 0), f"{nombre} ({score:.2f})"
                if nombre not in detectados or (now - detectados[nombre]).seconds > 30:
                    detectados[nombre] = now
                    print(f"✓ ENTRADA: {nombre} - {now.strftime('%H:%M:%S')}")
            else:
                color, label = (0, 0, 255), "Desconocido"
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f} | Empleados: {len(empleados)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Asistencia', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            nombre = input("Nombre: ").strip()
            if nombre:
                ret, frame = cap.read()
                faces = app.get(frame)
                if len(faces) == 1:
                    empleados.append({'nombre': nombre, 'embedding': faces[0].embedding})
                    guardar_db(empleados)
                    print(f"✓ {nombre} registrado!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
