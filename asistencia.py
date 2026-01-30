import cv2
import numpy as np
import json
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# Configuración
DB_FILE = '/data/empleados.json'
LOGS_FILE = '/data/asistencia_log.json'
THRESHOLD = 0.4  # Similitud mínima para reconocer

# Inicializar detector
print("Cargando modelo...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Modelo cargado!")

def cargar_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            # Convertir listas a numpy arrays
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
    print(f"[{log['timestamp']}] {tipo}: {nombre}")

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

def registrar_empleado(nombre):
    empleados = cargar_db()
    cap = cv2.VideoCapture(0)
    print(f"Registrando a {nombre}... Mira a la cámara.")
    
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
            print(f"Error: Se detectaron {len(faces)} caras. Debe haber exactamente 1.")
    cap.release()

def monitorear():
    empleados = cargar_db()
    if not empleados:
        print("No hay empleados registrados. Usa: registrar_empleado('Nombre')")
        return
    
    cap = cv2.VideoCapture(0)
    print(f"Monitoreando... {len(empleados)} empleados en DB. Ctrl+C para salir.")
    
    detectados_recientes = {}  # Evitar duplicados
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            faces = app.get(frame)
            
            for face in faces:
                nombre, score = reconocer(face.embedding, empleados)
                
                if nombre:
                    ahora = datetime.now()
                    # Evitar registrar la misma persona en menos de 30 seg
                    if nombre in detectados_recientes:
                        diff = (ahora - detectados_recientes[nombre]).seconds
                        if diff < 30:
                            continue
                    
                    detectados_recientes[nombre] = ahora
                    registrar_log(nombre, "ENTRADA")
                    print(f"✓ {nombre} detectado (score: {score:.2f})")
    
    except KeyboardInterrupt:
        print("\nMonitoreo detenido.")
    
    cap.release()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "registrar" and len(sys.argv) > 2:
            registrar_empleado(sys.argv[2])
        elif sys.argv[1] == "monitorear":
            monitorear()
        else:
            print("Uso:")
            print("  python asistencia.py registrar 'Nombre'")
            print("  python asistencia.py monitorear")
    else:
        print("Uso:")
        print("  python asistencia.py registrar 'Nombre'")
        print("  python asistencia.py monitorear")
